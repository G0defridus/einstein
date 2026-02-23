import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import requests
import xml.etree.ElementTree as ET
from datetime import timedelta

# Pagina instellingen
st.set_page_config(page_title="Energy Hedge & Financial Optimizer 8.0", layout="wide")

st.title("⚡ Energy Hedge & Financial Optimizer 8.0")

# --- ENTSO-E API FUNCTIES ---
@st.cache_data
def fetch_entsoe_prices(api_key, start_date, end_date, area_code="10YNL31N1001A590"):
    """
    Haalt Day-Ahead Prices op van ENTSO-E API.
    Area Code default = NL (10YNL31N1001A590)
    """
    # Datums formatteren voor API (YYYYMMDDHHMM)
    # We vragen iets ruimere periode op om tijdzone issues te dekken
    period_start = (start_date - timedelta(days=1)).strftime('%Y%m%d0000')
    period_end = (end_date + timedelta(days=1)).strftime('%Y%m%d2300')
    
    url = "https://web-api.tp.entsoe.eu/api"
    params = {
        "securityToken": api_key,
        "documentType": "A44",  # Price Document
        "in_Domain": area_code,
        "out_Domain": area_code,
        "periodStart": period_start,
        "periodEnd": period_end
    }
    
    try:
        response = requests.get(url, params=params)
        if response.status_code != 200:
            return None, f"HTTP Error {response.status_code}"
        
        # XML Parsen
        root = ET.fromstring(response.content)
        ns = {'ns': 'urn:iec62325.351:tc57wg16:451-3:publicationdocument:7:0'}
        
        prices = []
        for time_series in root.findall('.//ns:TimeSeries', ns):
            # Check resolutie (moet PT60M zijn voor day-ahead, soms kwartier)
            # Voor nu gaan we uit van standaard uurprijzen
            start_str = time_series.find('.//ns:timeInterval/ns:start', ns).text
            start_dt = pd.to_datetime(start_str)
            
            for point in time_series.findall('.//ns:Point', ns):
                position = int(point.find('ns:position', ns).text)
                price = float(point.find('ns:price.amount', ns).text)
                
                # Tijdstip berekenen (Positie 1 = starttijd)
                # Let op: dit is vaak UTC
                point_time = start_dt + timedelta(hours=position-1)
                prices.append({'Date': point_time, 'EPEX_Price': price})
                
        if not prices:
            return None, "Geen prijzen gevonden in XML."
            
        df_prices = pd.DataFrame(prices)
        df_prices['Date'] = pd.to_datetime(df_prices['Date'], utc=True)
        
        # Omzetten naar lokale tijd (Europe/Amsterdam) en indexeren
        df_prices = df_prices.set_index('Date').dt.tz_convert('Europe/Amsterdam').tz_localize(None)
        
        # Resamplen naar kwartieren (Forward Fill: uurprijs geldt voor 4 kwartieren)
        df_prices = df_prices.resample('15min').ffill()
        
        return df_prices, None
        
    except Exception as e:
        return None, str(e)

# --- DEEL 1: PROFIEL LOGICA ---
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

# --- DEEL 2: GUI & LOGIC ---

st.sidebar.header("1. Data Input")
input_mode = st.sidebar.radio("Input Type", ["Ruwe Aansluitingen (CSV)", "Reeds Geaggregeerd (CSV)"])
uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

df_hedge = None

if uploaded_file is not None:
    if input_mode == "Ruwe Aansluitingen (CSV)":
        try:
            df_agg, mapping = process_raw_connections(uploaded_file)
            with st.expander("ℹ️ Resultaat Analyse", expanded=True):
                c1, c2, c3 = st.columns(3)
                counts = pd.Series(mapping.values()).value_counts()
                c1.metric("Consumers", counts.get('Consumer', 0))
                c2.metric("Prosumers", counts.get('Prosumer', 0))
                c3.metric("Producers", counts.get('Producer', 0))
            df_hedge = df_agg.reset_index()
            cols = list(df_hedge.columns); cols[0] = 'Date'; df_hedge.columns = cols
        except Exception as e:
            st.error(f"Fout: {e}")
            st.stop()
    else: 
        try:
            df_hedge = pd.read_csv(uploaded_file, sep=';', decimal=',')
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
            st.error(f"Fout: {e}")
            st.stop()

if df_hedge is not None:
    df = df_hedge.copy()
    if not pd.api.types.is_datetime64_any_dtype(df['Date']):
        df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
    df = df.sort_values('Date').drop_duplicates('Date').set_index('Date').asfreq('15min').ffill().reset_index()

    for col in ['Consumer', 'Prosumer', 'Producer', 'Total']:
        if col in df.columns: df[f'{col}_MW'] = (df[col] * 4) / 1000
        else: df[f'{col}_MW'] = 0.0
    
    df['is_peak'] = (df['Date'].dt.weekday < 5) & (df['Date'].dt.hour >= 8) & (df['Date'].dt.hour < 20)
    df['Quarter'] = df['Date'].dt.quarter

    # --- API SECTIE ---
    st.sidebar.markdown("---")
    st.sidebar.header("2. Financiële Data (ENTSO-E)")
    
    api_key = st.sidebar.text_input("API Key", value="af129bcc-226c-4c4b-bfe1-dc7d9b5a5217", type="password")
    use_ref_year = st.sidebar.checkbox("Forceer referentiejaar 2024?", value=False, help="Vink aan als je 2025 data hebt maar nog geen EPEX prijzen bestaan.")
    
    # Prijzen ophalen
    start_date = df['Date'].min()
    end_date = df['Date'].max()
    
    # Logic voor referentiejaar (als data in toekomst ligt)
    if use_ref_year or start_date.year > pd.Timestamp.now().year:
        fetch_start = start_date.replace(year=2024)
        fetch_end = end_date.replace(year=2024)
        st.sidebar.info(f"Ophalen prijzen 2024 ({fetch_start.date()} - {fetch_end.date()}) en projecteren op dataset.")
    else:
        fetch_start = start_date
        fetch_end = end_date
    
    if 'epex_prices' not in st.session_state:
        st.session_state['epex_prices'] = None

    if st.sidebar.button("🔄 Haal EPEX Prijzen op"):
        with st.spinner("Verbinding maken met ENTSO-E..."):
            prices, error = fetch_entsoe_prices(api_key, fetch_start, fetch_end)
            if prices is not None:
                st.session_state['epex_prices'] = prices
                st.sidebar.success("Prijzen opgehaald!")
            else:
                st.sidebar.error(f"Fout: {error}")

    # Merge Prijzen
    df_fin = df.copy()
    if st.session_state['epex_prices'] is not None:
        prices = st.session_state['epex_prices'].copy()
        
        # Als we referentiejaar gebruiken, moeten we de 'prices' index verschuiven naar het jaar van de dataset
        if use_ref_year or start_date.year > pd.Timestamp.now().year:
            # Verschuif prijzen van 2024 terug naar het jaar van de dataset (bijv 2025)
            # Simpele manier: vervang jaar in index
            target_year = start_date.year
            # Let op schrikkeljaren: dit is 'tricky', we doen een eenvoudige merge op maand-dag-tijd
            prices['MatchKey'] = prices.index.strftime('%m-%d %H:%M')
            df_fin['MatchKey'] = df_fin['Date'].dt.strftime('%m-%d %H:%M')
            
            # Alleen de prijs kolom mergen
            df_fin = pd.merge(df_fin, prices[['EPEX_Price', 'MatchKey']], on='MatchKey', how='left')
        else:
            # Gewoon op datum mergen
            df_fin = pd.merge(df_fin, prices[['EPEX_Price']], left_on='Date', right_index=True, how='left')
            
        # Vul gaten (bijv. zomertijd)
        df_fin['EPEX_Price'] = df_fin['EPEX_Price'].fillna(method='ffill').fillna(80.0) # Fallback 80 euro
    else:
        df_fin['EPEX_Price'] = 80.0 # Default dummy prijs

    # --- HEDGE CONFIG ---
    st.sidebar.markdown("---")
    st.sidebar.header("3. Hedge & Prijzen")
    
    profile_choice = st.sidebar.selectbox("Kies Profiel", ["Consumer", "Prosumer", "Producer", "Total"])
    strategy_period = st.sidebar.radio("Periode", ["Per Jaar", "Per Kwartaal"])
    p_mw_col = f'{profile_choice}_MW'

    # Financiële Inputs
    st.sidebar.markdown("**Hedge Prijzen (Forward)**")
    c_h1, c_h2 = st.sidebar.columns(2)
    price_base = c_h1.number_input("Base Prijs (€/MWh)", value=100.0, step=5.0)
    price_peak = c_h2.number_input("Peak Prijs (€/MWh)", value=120.0, step=5.0)

    if 'slider_values' not in st.session_state: st.session_state['slider_values'] = {}

    # --- CALC FUNCTIES ---
    def calculate_metrics(sub_df, base, peak_add):
        hedge = base + (sub_df['is_peak'] * peak_add)
        prof = sub_df[p_mw_col]
        vol_hedge = hedge.sum() * 0.25
        vol_prof = prof.sum() * 0.25
        diff = hedge - prof
        over_hedge_mwh = diff[diff > 0].sum() * 0.25
        over_pct = (over_hedge_mwh / vol_prof * 100) if vol_prof != 0 else 0
        return vol_hedge, vol_prof, over_pct

    def find_optimal_mw(sub_df, target_over_pct_limit=None, percent_volume_target=None):
        if percent_volume_target is not None:
            off_peak_mean = sub_df.loc[~sub_df['is_peak'], p_mw_col].mean()
            peak_mean = sub_df.loc[sub_df['is_peak'], p_mw_col].mean()
            if pd.isna(off_peak_mean): off_peak_mean = 0.0
            if pd.isna(peak_mean): peak_mean = 0.0
            b = round(off_peak_mean * (percent_volume_target / 100.0), 1)
            p_add = round((peak_mean * (percent_volume_target / 100.0)) - b, 1)
            return b, p_add
        best_b, best_p = 0.0, 0.0
        for pct in range(150, 0, -1): 
            b_try, p_try = find_optimal_mw(sub_df, percent_volume_target=pct)
            _, _, over_pct = calculate_metrics(sub_df, b_try, p_try)
            if target_over_pct_limit is not None and over_pct <= target_over_pct_limit:
                best_b, best_p = b_try, p_try
                break 
        return best_b, best_p

    # --- STRATEGIE KNOPPEN ---
    def apply_strategy(strat_name):
        periods = [0] if strategy_period == "Per Jaar" else [1, 2, 3, 4]
        for q in periods:
            sub_df = df_fin if q == 0 else df_fin[df_fin['Quarter'] == q]
            if strat_name == "0%_sell": b, p = find_optimal_mw(sub_df, target_over_pct_limit=0.1)
            elif strat_name == "5%_sell": b, p = find_optimal_mw(sub_df, target_over_pct_limit=5.0)
            elif strat_name == "80%_cov": b, p = find_optimal_mw(sub_df, percent_volume_target=80)
            elif strat_name == "100%_cov": b, p = find_optimal_mw(sub_df, percent_volume_target=100)
            elif strat_name == "110%_cov": b, p = find_optimal_mw(sub_df, percent_volume_target=110)
            key_b = f'slider_b_yr' if q == 0 else f'slider_b_q{q}'
            key_p = f'slider_p_yr' if q == 0 else f'slider_p_q{q}'
            st.session_state[key_b] = float(b)
            st.session_state[key_p] = float(p)

    st.sidebar.markdown("Kies Strategie:")
    c1, c2 = st.sidebar.columns(2)
    if c1.button("🛡️ 0% Sell"): apply_strategy("0%_sell")
    if c2.button("🎯 Max 5% Sell"): apply_strategy("5%_sell")
    c3, c4 = st.sidebar.columns(2)
    if c3.button("📉 Basis 80%"): apply_strategy("80%_cov")
    if c4.button("⚖️ Balans 100%"): apply_strategy("100%_cov")
    if st.sidebar.button("📈 Over-Hedge 110%", use_container_width=True): apply_strategy("110%_cov")

    # --- SLIDERS ---
    st.sidebar.markdown("---")
    df_fin['Current_Hedge_MW'] = 0.0
    curr_min = df_fin[p_mw_col].min(); curr_max = df_fin[p_mw_col].max()
    slider_min = float(np.floor(curr_min * 1.5 - 1)); slider_max = float(np.ceil(curr_max * 1.5 + 1))
    if slider_max < slider_min: slider_max = slider_min + 10.0

    if strategy_period == "Per Jaar":
        if 'slider_b_yr' not in st.session_state:
            def_b, def_p = find_optimal_mw(df_fin, percent_volume_target=100)
            st.session_state['slider_b_yr'] = float(def_b); st.session_state['slider_p_yr'] = float(def_p)
        b_yr = st.sidebar.slider("Base (Jaar)", slider_min, slider_max, key="slider_b_yr", step=0.1)
        p_yr = st.sidebar.slider("Peak (Jaar)", slider_min, slider_max, key="slider_p_yr", step=0.1)
        df_fin['Current_Hedge_MW'] = b_yr + (df_fin['is_peak'] * p_yr)
    else:
        for q in [1, 2, 3, 4]:
            st.sidebar.markdown(f"**Kwartaal {q}**")
            q_mask = df_fin['Quarter'] == q
            if f'slider_b_q{q}' not in st.session_state:
                def_b, def_p = find_optimal_mw(df_fin[q_mask], percent_volume_target=100)
                st.session_state[f'slider_b_q{q}'] = float(def_b); st.session_state[f'slider_p_q{q}'] = float(def_p)
            c1, c2 = st.sidebar.columns(2)
            b_q = c1.slider(f"Q{q} Base", slider_min, slider_max, key=f"slider_b_q{q}", step=0.1)
            p_q = c2.slider(f"Q{q} Peak", slider_min, slider_max, key=f"slider_p_q{q}", step=0.1)
            df_fin.loc[q_mask, 'Current_Hedge_MW'] = b_q + (df_fin.loc[q_mask, 'is_peak'] * p_q)

    # --- FINANCIËLE BEREKENING ---
    df_fin['Profile_MWh'] = df_fin[p_mw_col] * 0.25
    df_fin['Hedge_MWh'] = df_fin['Current_Hedge_MW'] * 0.25
    
    # Kosten Hedge (Fixed Price)
    # We moeten weten wat Base en wat Peak volume is voor de prijs
    # Base Volume = Base MW * 0.25 (elk kwartier)
    # Peak Volume = Peak MW * 0.25 (alleen tijdens peak uren)
    
    # Omdat sliders dynamisch zijn per kwartaal, moeten we itereren of slim vectoriseren
    # Voor nu (versimpeld): We nemen aan dat Current_Hedge_MW correct is opgebouwd
    # We moeten de Base en Peak MW componenten weten om de prijs toe te passen.
    # Trucje: Als is_peak False is, is alles Base. Als True, is Base de waarde van de niet-peak uren (of slider).
    # Beter: we reconstrueren de Base/Peak volumes uit de sliders.
    
    # We maken hulpkolommen 'Hedge_Base_MW' en 'Hedge_Peak_MW'
    df_fin['Hedge_Base_MW'] = 0.0
    df_fin['Hedge_Peak_MW'] = 0.0
    
    if strategy_period == "Per Jaar":
        df_fin['Hedge_Base_MW'] = st.session_state['slider_b_yr']
        df_fin['Hedge_Peak_MW'] = np.where(df_fin['is_peak'], st.session_state['slider_p_yr'], 0.0)
    else:
        for q in [1, 2, 3, 4]:
            q_mask = df_fin['Quarter'] == q
            df_fin.loc[q_mask, 'Hedge_Base_MW'] = st.session_state[f'slider_b_q{q}']
            df_fin.loc[q_mask & df_fin['is_peak'], 'Hedge_Peak_MW'] = st.session_state[f'slider_p_q{q}']
            
    df_fin['Cost_Hedge'] = (df_fin['Hedge_Base_MW'] * 0.25 * price_base) + (df_fin['Hedge_Peak_MW'] * 0.25 * price_peak)
    
    # Kosten Spot (Unhedged)
    # Restprofiel = Verbruik - Hedge
    # Positief = Tekort (Kopen tegen EPEX)
    # Negatief = Overschot (Verkopen tegen EPEX)
    df_fin['Residual_MWh'] = df_fin['Profile_MWh'] - df_fin['Hedge_MWh']
    df_fin['Cost_Spot'] = df_fin['Residual_MWh'] * df_fin['EPEX_Price']
    
    df_fin['Total_Cost'] = df_fin['Cost_Hedge'] + df_fin['Cost_Spot']
    
    # KPI's
    total_prof = df_fin['Profile_MWh'].sum()
    total_cost = df_fin['Total_Cost'].sum()
    avg_price = total_cost / total_prof if total_prof != 0 else 0
    spot_exposure = df_fin['Cost_Spot'].sum()

    st.subheader("💰 Financiële Impact")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Totale Energiekosten", f"€ {total_cost:,.0f}")
    col2.metric("Gem. Prijs (All-in)", f"€ {avg_price:.2f} /MWh")
    col3.metric("Kosten uit Hedge", f"€ {df_fin['Cost_Hedge'].sum():,.0f}", help="Vaste kosten o.b.v. Base/Peak prijzen.")
    col4.metric("Spot Resultaat (Rest)", f"€ {spot_exposure:,.0f}", delta_color="inverse", help="Kosten/Opbrengsten van het restprofiel op de EPEX.")

    # Grafieken
    st.markdown("---")
    tab1, tab2 = st.tabs(["⚡ Volumes & Hedge", "💶 Kosten & Spot"])
    
    with tab1:
        st.caption("Blauw = Verbruik | Oranje = Hedge Blokken")
        chart_vol = alt.Chart(df_fin.melt(id_vars=['Date'], value_vars=[p_mw_col, 'Current_Hedge_MW'], var_name='Type', value_name='MW')).mark_line(interpolate='step-after').encode(
            x='Date:T', y='MW:Q', color='Type:N'
        ).properties(height=300)
        st.altair_chart(chart_vol, use_container_width=True)
        
    with tab2:
        st.caption("Grijze lijn = EPEX Prijs | Rode vlakken = Spot Kosten | Groene vlakken = Spot Opbrengst")
        # Voor visualisatie aggregeren we even naar dag, anders wordt het traag/lelijk
        df_day = df_fin.resample('D', on='Date').agg({'Cost_Spot': 'sum', 'EPEX_Price': 'mean'}).reset_index()
        
        base = alt.Chart(df_day).encode(x='Date:T')
        bar = base.mark_bar().encode(y='Cost_Spot:Q', color=alt.condition(alt.datum.Cost_Spot > 0, alt.value("red"), alt.value("green")))
        line = base.mark_line(color='gray').encode(y='EPEX_Price:Q')
        
        st.altair_chart((bar + line).resolve_scale(y='independent'), use_container_width=True)

else:
    st.info("👆 Upload een bestand om te beginnen.")
