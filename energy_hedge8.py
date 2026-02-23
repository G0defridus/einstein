import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import os
import glob

# Pagina instellingen
st.set_page_config(page_title="Energy Hedge Optimizer 8.5", layout="wide")

st.title("⚡ Energy Hedge Optimizer 8.5")

# --- DOCUMENTATIE BLOK (Inklapbaar) ---
with st.expander("📘 Lees mij: Achtergrond en Methodiek (Klik om te openen)", expanded=False):
    st.markdown("""
    ### 1. Van Ruwe Data naar Profiel
    Het model analyseert slimme meter data om te bepalen of een aansluiting een **Consumer**, **Prosumer** of **Producer**.
    * **Stap 1 (Winterprofiel):** We kijken naar Jan/Nov/Dec om het basisverbruik te bepalen.
    * **Stap 2 (Zon Detectie):** We vergelijken de zomer met de winter. Het verschil is de bruto zonne-opwek.
    * **Stap 3 (Beslisboom):** Op basis van ratio's wordt de aansluiting toegekend aan een profiel.
    
    ### 2. Hedge Strategie
    We kopen in op de groothandelsmarkt in blokken van **0,1 MW**.
    * **Base:** 24/7 vast vermogen.
    * **Peak:** Extra vermogen op werkdagen (08:00 - 20:00).
    * **Total:** Inkoop voor het samengestelde portfolio-profiel (salderen).
    
    ### 3. Financiële Waardering (Blokken & Spot)
    * **Blokken:** Je vult je vaste contractprijzen in voor de Base- en Peak-blokken. Dit bepaalt je vaste kosten.
    * **Spotmarkt (EPEX):** Via de ENTSO-E API wordt de Day-Ahead prijs ingeladen. Tekorten (Under-hedge) worden tegen deze prijs ingekocht, overschotten (Over-hedge) worden tegen deze prijs verkocht.
    * **Totale Energiekosten:** Kosten Blokken - Netto Spot Resultaat. Dit geeft de daadwerkelijke integrale MWh-prijs over het jaar.
    """)

# --- DEEL 1: LOGICA VOOR CATEGORISATIE ---
def calculate_winter_profile(df):
    winter_mask = df.index.month.isin([1, 11, 12])
    winter_df = df[winter_mask]
    if winter_df.empty: return df.groupby(df.index.time).mean()
    return winter_df.groupby(winter_df.index.time).mean()

def estimate_gross_solar_robust(df, connection_col, winter_profile):
    col_data = df[connection_col]
    w_prof = winter_profile[connection_col]
    night_mask = (df.index.hour < 6) | (df.index.hour >= 23)
    daily_night_avg = df.loc[night_mask, connection_col].resample('D').mean()
    night_times = [t for t in w_prof.index if t.hour < 6 or t.hour >= 23]
    base_night_avg = w_prof.loc[night_times].mean()
    if base_night_avg < 0.05: base_night_avg = 0.05
    daily_scaling = daily_night_avg / base_night_avg
    daily_scaling = daily_scaling.clip(0.2, 5.0)
    dates = pd.Series(df.index.normalize(), index=df.index)
    scaling_series = dates.map(daily_scaling).ffill().bfill()
    base_load_series = df.index.map(lambda x: w_prof.loc[x.time()]) 
    base_load_series = pd.Series(base_load_series, index=df.index)
    expected_load = base_load_series * scaling_series
    solar_behind_meter = expected_load - col_data
    is_daylight = (df.index.hour >= 8) & (df.index.hour <= 20)
    solar_behind_meter = solar_behind_meter.where(is_daylight, 0).clip(lower=0)
    actual_export = col_data.clip(upper=0).abs()
    return solar_behind_meter + actual_export

@st.cache_data
def process_raw_connections(file):
    try:
        df = pd.read_csv(file, sep=';', decimal=',', index_col=0, parse_dates=True, dayfirst=True)
        if not isinstance(df.index, pd.DatetimeIndex): raise ValueError
    except:
        df = pd.read_csv(file, sep=';', decimal=',')
        df['Date'] = pd.to_datetime(df.iloc[:, 0], dayfirst=True)
        df = df.set_index('Date')

    df = df.select_dtypes(include=[np.number])
    winter_profile = calculate_winter_profile(df)
    
    connection_cols = df.columns
    estimated_volumes = {}
    gross_prod_dict = {}
    my_bar = st.progress(0, text="Analyseren aansluitingen...")
    total_cols = len(connection_cols)
    for i, col in enumerate(connection_cols):
        gross_series = estimate_gross_solar_robust(df, col, winter_profile)
        gross_prod_dict[col] = gross_series
        estimated_volumes[col] = gross_series.sum()
        if i % max(1, int(total_cols/10)) == 0:
            my_bar.progress((i + 1) / total_cols, text=f"Analyseren: {col}")
    my_bar.empty()
    
    gross_production_df = pd.DataFrame(gross_prod_dict, index=df.index)
    categories = {}
    for col in connection_cols:
        gross_vol = estimated_volumes[col]
        if gross_vol > 1000: 
            total_import = df[col][df[col] > 0].sum()
            if total_import < 0.2 * gross_vol: categories[col] = 'Producer'
            else: categories[col] = 'Prosumer'
        else: categories[col] = 'Consumer'
            
    hours = np.arange(24)
    solar_curve_ideal = np.exp(-((hours - 13)**2) / (2 * 2.5**2))
    solar_curve_ideal[hours < 6] = 0
    solar_curve_ideal[hours > 21] = 0
    final_mapping = {}
    for col in connection_cols:
        cat = categories[col]
        if cat == 'Prosumer':
            daily_avg = gross_production_df[col].groupby(gross_production_df[col].index.hour).mean()
            daily_avg = daily_avg.reindex(range(24), fill_value=0)
            corr = 0
            if np.std(daily_avg) > 0 and np.std(solar_curve_ideal) > 0:
                corr = np.corrcoef(daily_avg, solar_curve_ideal)[0, 1]
            if corr < 0.85: final_mapping[col] = 'Consumer'
            else: final_mapping[col] = 'Prosumer'
        else:
            final_mapping[col] = cat
            
    cat_consumer = [c for c, cat in final_mapping.items() if cat == 'Consumer']
    cat_prosumer = [c for c, cat in final_mapping.items() if cat == 'Prosumer']
    cat_producer = [c for c, cat in final_mapping.items() if cat == 'Producer']
    
    agg_df = pd.DataFrame(index=df.index)
    agg_df['Consumer'] = df[cat_consumer].sum(axis=1) if cat_consumer else 0.0
    agg_df['Prosumer'] = df[cat_prosumer].sum(axis=1) if cat_prosumer else 0.0
    agg_df['Producer'] = df[cat_producer].sum(axis=1) if cat_producer else 0.0
    agg_df['Total'] = agg_df['Consumer'] + agg_df['Prosumer'] + agg_df['Producer']
    return agg_df, final_mapping

# --- EPEX API FUNCTIE ---
@st.cache_data(show_spinner="Prijzen downloaden via ENTSO-E API...")
def fetch_epex_prices(api_key, start_date, end_date):
    try:
        from entsoe import EntsoePandasClient
        client = EntsoePandasClient(api_key=api_key)
        start = pd.Timestamp(start_date).tz_localize('Europe/Amsterdam')
        end = (pd.Timestamp(end_date) + pd.Timedelta(days=1)).tz_localize('Europe/Amsterdam')
        
        ts = client.query_day_ahead_prices('NL', start=start, end=end)
        df_epex = ts.to_frame('EPEX_EUR_MWh')
        
        df_epex['Date_Hour'] = df_epex.index.tz_localize(None)
        df_epex = df_epex.drop_duplicates(subset='Date_Hour', keep='first')
        return df_epex
    except Exception as e:
        return str(e)

# --- DEEL 2: UI & DATA PREP ---
st.sidebar.header("1. Data Input")
input_mode = st.sidebar.radio("Input Type", ["Ruwe Aansluitingen (CSV)", "Reeds Geaggregeerd (CSV)"])
uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

df_hedge = None

if uploaded_file is not None:
    if input_mode == "Ruwe Aansluitingen (CSV)":
        try:
            df_agg, mapping = process_raw_connections(uploaded_file)
            with st.expander("ℹ️ Resultaat Analyse & Categorisatie", expanded=True):
                c1, c2, c3 = st.columns(3)
                counts = pd.Series(mapping.values()).value_counts()
                c1.metric("Consumers", counts.get('Consumer', 0))
                c2.metric("Prosumers", counts.get('Prosumer', 0))
                c3.metric("Producers", counts.get('Producer', 0))
            
            df_hedge = df_agg.reset_index()
            cols = list(df_hedge.columns)
            cols[0] = 'Date'
            df_hedge.columns = cols
        except Exception as e:
            st.error(f"Fout bij verwerken ruwe data: {e}")
            st.stop()
    else: 
        try:
            df_hedge = pd.read_csv(uploaded_file, sep=';', decimal=',')
            if 'Date' not in df_hedge.columns:
                for c in ['Datum', 'Tijd', 'Time', 'date', 'time']:
                    if c in df_hedge.columns:
                        df_hedge.rename(columns={c: 'Date'}, inplace=True)
                        break
            if 'Date' not in df_hedge.columns:
                cols = list(df_hedge.columns); cols[0] = 'Date'; df_hedge.columns = cols

            for col in ['Consumer', 'Prosumer', 'Producer', 'Total']:
                if col in df_hedge.columns:
                    df_hedge[col] = pd.to_numeric(df_hedge[col].astype(str).str.replace(',', '.'), errors='coerce')
            
            if 'Total' not in df_hedge.columns:
                cols_to_sum = [c for c in ['Consumer', 'Prosumer', 'Producer'] if c in df_hedge.columns]
                df_hedge['Total'] = df_hedge[cols_to_sum].sum(axis=1) if cols_to_sum else 0.0
            
            df_hedge['Date'] = pd.to_datetime(df_hedge['Date'], dayfirst=True)
        except Exception as e:
            st.error(f"Fout bij inlezen bestand: {e}")
            st.stop()

# --- HEDGE LOGICA & BEREKENINGEN ---
if df_hedge is not None:
    df = df_hedge.copy()
    if not pd.api.types.is_datetime64_any_dtype(df['Date']):
        df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)

    df = df.sort_values('Date').drop_duplicates(subset='Date', keep='first')
    df = df.set_index('Date').asfreq('15min').ffill().reset_index()

    for col in ['Consumer', 'Prosumer', 'Producer', 'Total']:
        if col in df.columns: df[f'{col}_MW'] = (df[col] * 4) / 1000
        else: df[f'{col}_MW'] = 0.0
    
    df['is_peak'] = (df['Date'].dt.weekday < 5) & (df['Date'].dt.hour >= 8) & (df['Date'].dt.hour < 20)
    df['Quarter'] = df['Date'].dt.quarter

    st.sidebar.markdown("---")
    st.sidebar.header("2. Hedge Configuratie")
    profile_choice = st.sidebar.selectbox("Kies Profiel", ["Consumer", "Prosumer", "Producer", "Total"])
    strategy_period = st.sidebar.radio("Periode", ["Per Jaar", "Per Kwartaal"])
    p_mw_col = f'{profile_choice}_MW'

    if 'slider_values' not in st.session_state: st.session_state['slider_values'] = {}

    def calculate_metrics(sub_df, base, peak_add):
        hedge = base + (sub_df['is_peak'] * peak_add)
        prof = sub_df[p_mw_col]
        vol_prof = prof.sum() * 0.25
        diff = hedge - prof
        over_hedge_mwh = diff[diff > 0].sum() * 0.25
        over_pct = (over_hedge_mwh / vol_prof * 100) if vol_prof != 0 else 0
        return over_pct

    def find_optimal_mw(sub_df, target_over_pct_limit=None, percent_volume_target=None):
        if percent_volume_target is not None:
            off_peak_mean = sub_df.loc[~sub_df['is_peak'], p_mw_col].mean()
            peak_mean = sub_df.loc[sub_df['is_peak'], p_mw_col].mean()
            b = round(off_peak_mean * (percent_volume_target / 100.0), 1) if not pd.isna(off_peak_mean) else 0.0
            p_add = round((peak_mean * (percent_volume_target / 100.0)) - b, 1) if not pd.isna(peak_mean) else 0.0
            return b, p_add

        best_b, best_p = 0.0, 0.0
        for pct in range(150, 0, -1): 
            b_try, p_try = find_optimal_mw(sub_df, percent_volume_target=pct)
            over_pct = calculate_metrics(sub_df, b_try, p_try)
            if target_over_pct_limit is not None and over_pct <= target_over_pct_limit:
                best_b, best_p = b_try, p_try
                break 
        return best_b, best_p

    # --- STRATEGIE BLOK ---
    st.sidebar.markdown("---")
    st.sidebar.header("3. Kies Strategie")

    def apply_strategy(strat_name, custom_pct=None):
        periods = [0] if strategy_period == "Per Jaar" else [1, 2, 3, 4]
        for q in periods:
            sub_df = df if q == 0 else df[df['Quarter'] == q]
            if strat_name == "5%_sell": b, p = find_optimal_mw(sub_df, target_over_pct_limit=5.0)
            elif strat_name == "10%_cov": b, p = find_optimal_mw(sub_df, percent_volume_target=10)
            elif strat_name == "100%_cov": b, p = find_optimal_mw(sub_df, percent_volume_target=100)
            elif strat_name == "custom_cov": b, p = find_optimal_mw(sub_df, percent_volume_target=custom_pct)
            
            st.session_state[f'slider_b_yr' if q == 0 else f'slider_b_q{q}'] = float(b)
            st.session_state[f'slider_p_yr' if q == 0 else f'slider_p_q{q}'] = float(p)

    def on_custom_hedge_change():
        apply_strategy("custom_cov", custom_pct=st.session_state.custom_hedge_pct)

    c1, c2, c3 = st.sidebar.columns(3)
    if c1.button("🎯 Max 5% Sell"): apply_strategy("5%_sell")
    if c2.button("📉 10% Hedge"): apply_strategy("10%_cov")
    if c3.button("⚖️ 100% Hedge"): apply_strategy("100%_cov")

    st.sidebar.slider(
        "% Hedge op Volume", 
        min_value=0, max_value=150, value=100, step=1, 
        key="custom_hedge_pct", 
        on_change=on_custom_hedge_change,
        help="Versleep deze slider om de dekking aan te passen. De Base & Peak volumes (MW) worden direct geüpdatet."
    )

    # --- Sliders MW (Fine-Tuning) ---
    st.sidebar.markdown("---")
    st.sidebar.subheader("4. Fine-tuning Volumes (MW)")
    
    df['Hedge_Base_MW'] = 0.0
    df['Hedge_Peak_MW'] = 0.0
    
    curr_min = df[p_mw_col].min()
    curr_max = df[p_mw_col].max()
    slider_min = float(np.floor(curr_min * 1.5 - 1))
    slider_max = float(np.ceil(curr_max * 1.5 + 1))
    if slider_max < slider_min: slider_max = slider_min + 10.0

    if strategy_period == "Per Jaar":
        if 'slider_b_yr' not in st.session_state:
            def_b, def_p = find_optimal_mw(df, percent_volume_target=100)
            st.session_state['slider_b_yr'] = float(def_b); st.session_state['slider_p_yr'] = float(def_p)
            
        b_yr = st.sidebar.slider("Base (Jaar)", slider_min, slider_max, key="slider_b_yr", step=0.1)
        p_yr = st.sidebar.slider("Peak (Jaar)", slider_min, slider_max, key="slider_p_yr", step=0.1)
        
        df['Hedge_Base_MW'] = b_yr
        df['Hedge_Peak_MW'] = p_yr * df['is_peak']
    else:
        for q in [1, 2, 3, 4]:
            st.sidebar.markdown(f"**Kwartaal {q}**")
            q_mask = df['Quarter'] == q
            if f'slider_b_q{q}' not in st.session_state:
                def_b, def_p = find_optimal_mw(df[q_mask], percent_volume_target=100)
                st.session_state[f'slider_b_q{q}'] = float(def_b); st.session_state[f'slider_p_q{q}'] = float(def_p)
            
            sc1, sc2 = st.sidebar.columns(2)
            b_q = sc1.slider(f"Q{q} Base", slider_min, slider_max, key=f"slider_b_q{q}", step=0.1)
            p_q = sc2.slider(f"Q{q} Peak", slider_min, slider_max, key=f"slider_p_q{q}", step=0.1)
            
            df.loc[q_mask, 'Hedge_Base_MW'] = b_q
            df.loc[q_mask, 'Hedge_Peak_MW'] = p_q * df.loc[q_mask, 'is_peak']
            
    df['Current_Hedge_MW'] = df['Hedge_Base_MW'] + df['Hedge_Peak_MW']

    # --- HELPER: ROBUUSTE ENDEX EXTRACTIE ---
    @st.cache_data
    def get_default_price(period="Jaar", q=None):
        # 1. Fallback realistische waarden (o.b.v. profiel 2024-2025)
        b_val, p_val = 80.0, 95.0
        if period == "Kwartaal":
            if q == 1: b_val, p_val = 90.0, 110.0
            elif q == 2: b_val, p_val = 65.0, 75.0
            elif q == 3: b_val, p_val = 70.0, 80.0
            elif q == 4: b_val, p_val = 85.0, 105.0

        try:
            # 2. Zoek dynamisch naar de juiste CSV file lokaal of in mappen
            if period == "Jaar":
                files = glob.glob("**/*jaar*endex*.csv", recursive=True) + glob.glob("**/*endex*jaar*.csv", recursive=True)
            else:
                files = glob.glob("**/*kwartaal*endex*.csv", recursive=True) + glob.glob("**/*endex*kwartaal*.csv", recursive=True)
                
            if not files: # Laatste redmiddel als glob faalt
                files = [f for f in os.listdir('.') if period.lower() in f.lower() and 'endex' in f.lower() and f.endswith('.csv')]
                
            if files:
                # ICE ENDEX export bevat 2 header rijen (CAL-x erboven). We checken dit en slaan rij 0 over als nodig.
                with open(files[0], 'r', encoding='utf-8', errors='ignore') as f:
                    lines = f.readlines()
                
                skip = 0
                if len(lines) > 1 and "Date" in lines[1] and "Base" in lines[1]:
                    skip = 1
                    
                df_px = pd.read_csv(files[0], sep=';', skiprows=skip)
                
                # Zorg dat de komma's punten worden (NL formaat)
                for c in df_px.columns:
                    if df_px[c].dtype == object:
                        df_px[c] = df_px[c].astype(str).str.replace(',', '.')

                # Identificeer de base en peak kolommen (P16hrs is ENDEX term voor Peak)
                base_cols = [c for c in df_px.columns if 'base' in c.lower()]
                peak_cols = [c for c in df_px.columns if 'p16' in c.lower() or 'peak' in c.lower()]
                
                # Omdat er meerdere verhandelde contracten in 1 file staan (CAL-25, CAL-26 etc)
                # Doorzoeken we ze van rechts naar links om de meest actuele, liquide data te vinden
                if base_cols and peak_cols:
                    found_b, found_p = False, False
                    for bc in reversed(base_cols):
                        s = pd.to_numeric(df_px[bc], errors='coerce').dropna()
                        if len(s) > 10:  # Moet enigszins liquide zijn
                            b_val = float(s.iloc[-1])
                            found_b = True
                            break
                            
                    for pc in reversed(peak_cols):
                        s = pd.to_numeric(df_px[pc], errors='coerce').dropna()
                        if len(s) > 10:
                            p_val = float(s.iloc[-1])
                            found_p = True
                            break
                            
                    # Als we in de kwartaal file zitten en het kwartaal kunnen matchen
                    if period == "Kwartaal" and found_b and found_p:
                        # Bereken de seizoensfactor o.b.v de fallbacks om de gevonden actuele prijs bij te schaven
                        # (Zodat Q2 logischerwijs goedkoper blijft dan Q1 als we de specifieke Q niet kunnen parsen)
                        gem_b = (90+65+70+85)/4
                        gem_p = (110+75+80+105)/4
                        if q == 1: b_val *= (90/gem_b); p_val *= (110/gem_p)
                        elif q == 2: b_val *= (65/gem_b); p_val *= (75/gem_p)
                        elif q == 3: b_val *= (70/gem_b); p_val *= (80/gem_p)
                        elif q == 4: b_val *= (85/gem_b); p_val *= (105/gem_p)
                        
        except Exception as e:
            pass # Als er wat dan ook mis gaat, faal geruisloos en gebruik de veilige fallbacks
            
        return round(b_val, 2), round(p_val, 2)

    # --- CONTRACTPRIJZEN BLOKKEN (MET AUTOMATISCHE PRE-FILLS) ---
    st.sidebar.markdown("---")
    st.sidebar.subheader("5. Contractprijzen (Inkoopblokken)")
    
    df['Price_Base'] = 0.0
    df['Price_Peak'] = 0.0
    
    if strategy_period == "Per Jaar":
        cp1, cp2 = st.sidebar.columns(2)
        def_b, def_p = get_default_price("Jaar")
        pr_b = cp1.number_input("Base Prijs (€/MWh)", value=def_b, step=1.0)
        pr_p = cp2.number_input("Peak Prijs (€/MWh)", value=def_p, step=1.0)
        df['Price_Base'] = pr_b
        df['Price_Peak'] = pr_p
    else:
        for q in [1, 2, 3, 4]:
            st.sidebar.markdown(f"**Prijzen Q{q}**")
            cp1, cp2 = st.sidebar.columns(2)
            def_b, def_p = get_default_price("Kwartaal", q)
            pr_b = cp1.number_input(f"Q{q} Base (€/MWh)", value=def_b, step=1.0, key=f"pr_b_q{q}")
            pr_p = cp2.number_input(f"Q{q} Peak (€/MWh)", value=def_p, step=1.0, key=f"pr_p_q{q}")
            q_mask = df['Quarter'] == q
            df.loc[q_mask, 'Price_Base'] = pr_b
            df.loc[q_mask, 'Price_Peak'] = pr_p

    # --- ENTSO-E INPUT ---
    st.sidebar.markdown("---")
    st.sidebar.header("6. Financiële Module (Spotmarkt EPEX)")
    use_epex = st.sidebar.checkbox("Laad Spotprijzen in via ENTSO-E", value=False)
    
    epex_loaded = False
    if use_epex:
        if "ENTSOE_API_KEY" not in st.secrets:
            st.sidebar.error("⚠️ Systeemconfiguratiefout: ENTSO-E API Key ontbreekt in de server instellingen (.streamlit/secrets.toml).")
        else:
            api_key = st.secrets["ENTSOE_API_KEY"]
            start_dt = df['Date'].min()
            end_dt = df['Date'].max()
            df_epex = fetch_epex_prices(api_key, start_dt, end_dt)
            
            if isinstance(df_epex, str):
                st.sidebar.error(f"⚠️ Fout bij ophalen EPEX: Controleer de API Key en zorg dat 'entsoe-py' in requirements.txt staat. Details: {df_epex}")
            else:
                df['Date_Hour'] = df['Date'].dt.floor('H')
                df = pd.merge(df, df_epex[['Date_Hour', 'EPEX_EUR_MWh']], on='Date_Hour', how='left')
                epex_loaded = True
                st.sidebar.success("EPEX Prijzen succesvol op de achtergrond gekoppeld!")

    if not epex_loaded:
        df['EPEX_EUR_MWh'] = 0.0

    # --- RESULTATEN BEREKENEN ---
    df['Profile_MWh'] = df[p_mw_col] * 0.25
    df['Hedge_MWh'] = df['Current_Hedge_MW'] * 0.25
    df['Used_Hedge_MWh'] = np.minimum(df['Hedge_MWh'], df['Profile_MWh']) 
    df['Over_Hedge_MWh'] = np.maximum(0, df['Hedge_MWh'] - df['Profile_MWh'])
    df['Under_Hedge_MWh'] = np.maximum(0, df['Profile_MWh'] - df['Hedge_MWh'])

    # Blok kosten = (MW * 0.25) * Prijs
    df['Cost_Hedge_Base_EUR'] = (df['Hedge_Base_MW'] * 0.25) * df['Price_Base']
    df['Cost_Hedge_Peak_EUR'] = (df['Hedge_Peak_MW'] * 0.25) * df['Price_Peak']
    df['Cost_Hedge_Total_EUR'] = df['Cost_Hedge_Base_EUR'] + df['Cost_Hedge_Peak_EUR']
    
    tot_hedge_eur = df['Cost_Hedge_Total_EUR'].sum()

    # Spotmarkt kosten
    df['Cost_Buy_EUR'] = df['Under_Hedge_MWh'] * df['EPEX_EUR_MWh'] if epex_loaded else 0.0
    df['Rev_Sell_EUR'] = df['Over_Hedge_MWh'] * df['EPEX_EUR_MWh'] if epex_loaded else 0.0
    df['Net_Spot_EUR'] = df['Rev_Sell_EUR'] - df['Cost_Buy_EUR'] if epex_loaded else 0.0

    total_prof = df['Profile_MWh'].sum()
    total_over = df['Over_Hedge_MWh'].sum()
    total_under = df['Under_Hedge_MWh'].sum()
    denom = total_prof if total_prof != 0 else 1.0
    
    # UI METRICS VOLUMES
    st.markdown("### 📊 Volume Balans")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Effectieve Dekking", f"{(df['Used_Hedge_MWh'].sum()/denom)*100:.1f}%")
    k2.metric("Totaal Verbruik", f"{total_prof:,.0f} MWh")
    k3.metric("Teveel (Over-hedge)", f"{total_over:,.0f} MWh", f"{(total_over/denom)*100:.1f}%", delta_color="inverse")
    k4.metric("Tekort (Under-hedge)", f"{total_under:,.0f} MWh", f"{(total_under/denom)*100:.1f}%", delta_color="inverse")

    # UI METRICS FINANCIEEL
    if epex_loaded:
        st.markdown("### 💶 Financiële Waardering Totaal (Blokken + Spotmarkt)")
        f1, f2, f3, f4 = st.columns(4)
        
        net_spot_eur = df['Net_Spot_EUR'].sum()
        tot_energy_cost = tot_hedge_eur - net_spot_eur 
        
        f1.metric("Kosten Inkoopblokken", f"€ {tot_hedge_eur:,.0f}", help="Vaste inkoopkosten o.b.v. de ingestelde prijzen in de zijbalk.")
        
        net_color = "normal" if net_spot_eur >= 0 else "inverse"
        f2.metric("Netto Spot Resultaat", f"€ {net_spot_eur:,.0f}", delta="Winst" if net_spot_eur >=0 else "Verlies", delta_color=net_color, help="Verkoopopbrengst minus Inkoopkosten op de spotmarkt.")
        
        f3.metric("Totale Energiekosten", f"€ {tot_energy_cost:,.0f}", help="Kosten Inkoopblokken MINUS Netto Spot Resultaat.")
        
        avg_cost = tot_energy_cost / denom if denom > 0 else 0
        f4.metric("Integrale Kostprijs", f"€ {avg_cost:.2f} / MWh", help="Totale Energiekosten gedeeld door je totale MWh verbruik.")
    else:
        st.markdown("### 💶 Financiële Waardering (Alleen Inkoopblokken)")
        f1, f2, f3 = st.columns(3)
        f1.metric("Kosten Inkoopblokken", f"€ {tot_hedge_eur:,.0f}", help="Vaste inkoopkosten o.b.v. de ingestelde prijzen in de zijbalk.")
        gem_blok = tot_hedge_eur / df['Hedge_MWh'].sum() if df['Hedge_MWh'].sum() > 0 else 0
        f2.metric("Gemiddelde Blokprijs", f"€ {gem_blok:.2f} / MWh")
        st.info("💡 Zet de EPEX Spotprijzen aan in de zijbalk om je volledige integrale kostprijs te bereken.")

    st.markdown("---")
    st.subheader("🔎 Seizoensanalyse (4 Representatieve Weken)")
    weeks = [
        {"name": "Februari", "start": "2025-02-03", "end": "2025-02-09"},
        {"name": "Mei",       "start": "2025-05-05", "end": "2025-05-11"},
        {"name": "Augustus",  "start": "2025-08-04", "end": "2025-08-10"},
        {"name": "November", "start": "2025-11-03", "end": "2025-11-09"}
    ]
    cols = st.columns(2) + st.columns(2)
    for i, week in enumerate(weeks):
        with cols[i]:
            st.caption(week['name'])
            if df['Date'].min() > pd.Timestamp(week['end']) or df['Date'].max() < pd.Timestamp(week['start']):
                st.info("Geen data")
                continue
            mask = (df['Date'] >= week['start']) & (df['Date'] <= pd.Timestamp(week['end']) + pd.Timedelta(days=1))
            chart_data = df.loc[mask].melt(id_vars=['Date'], value_vars=[p_mw_col, 'Current_Hedge_MW'], var_name='Type', value_name='MW')
            chart_data['Type'] = chart_data['Type'].replace({p_mw_col: 'Verbruik', 'Current_Hedge_MW': 'Hedge'})
            c = alt.Chart(chart_data).mark_line(interpolate='step-after').encode(
                x=alt.X('Date:T', axis=alt.Axis(format='%a %H:%M', title=None)),
                y=alt.Y('MW:Q', title=None), color=alt.Color('Type:N', legend=alt.Legend(orient='bottom', title=None))
            ).properties(height=180)
            st.altair_chart(c, use_container_width=True)

    st.markdown("---")
    st.subheader("📊 Kwartaal Balans")
    
    q_stats = df.groupby('Quarter').apply(lambda x: pd.Series({
        'Verbruik (MWh)': x['Profile_MWh'].sum(),
        'Dekking %': (x['Used_Hedge_MWh'].sum() / x['Profile_MWh'].sum() * 100) if x['Profile_MWh'].sum() != 0 else 0,
        'Teveel (MWh)': x['Over_Hedge_MWh'].sum(),
        'Tekort (MWh)': x['Under_Hedge_MWh'].sum(),
        'Blokken Kosten (€)': x['Cost_Hedge_Total_EUR'].sum(),
        'EPEX Gem. (€/MWh)': x['EPEX_EUR_MWh'].mean() if epex_loaded else 0,
        'Netto Spot (€)': x['Net_Spot_EUR'].sum() if epex_loaded else 0,
        'Totale Kosten (€)': (x['Cost_Hedge_Total_EUR'].sum() - x['Net_Spot_EUR'].sum()) if epex_loaded else x['Cost_Hedge_Total_EUR'].sum(),
        'Gem. Prijs (€/MWh)': ((x['Cost_Hedge_Total_EUR'].sum() - x['Net_Spot_EUR'].sum()) / x['Profile_MWh'].sum()) if epex_loaded and x['Profile_MWh'].sum() != 0 else (x['Cost_Hedge_Total_EUR'].sum() / x['Profile_MWh'].sum() if x['Profile_MWh'].sum() != 0 else 0)
    }))

    format_dict = {
        'Verbruik (MWh)': "{:,.0f}", 'Dekking %': "{:.1f}%", 'Teveel (MWh)': "{:,.0f}", 'Tekort (MWh)': "{:,.0f}",
        'Blokken Kosten (€)': "€ {:,.0f}", 'Totale Kosten (€)': "€ {:,.0f}", 'Gem. Prijs (€/MWh)': "€ {:.2f}"
    }
    if epex_loaded:
        format_dict.update({'EPEX Gem. (€/MWh)': "€ {:.2f}", 'Netto Spot (€)': "€ {:,.0f}"})
    else:
        q_stats = q_stats.drop(columns=['EPEX Gem. (€/MWh)', 'Netto Spot (€)'])

    st.dataframe(q_stats.style.format(format_dict), use_container_width=True)

    csv_dl = df.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Download Detail Data (CSV)", csv_dl, "hedge_resultaten.csv", "text/csv")
else:
    st.info("👆 Upload een bestand om te beginnen.")
