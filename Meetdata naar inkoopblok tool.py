# -*- coding: utf-8 -*-
"""
Created on Mon Feb  9 08:21:48 2026

@author: FritsMaas
"""
import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import os

# Pagina instellingen
st.set_page_config(page_title="Energy Hedge & Profile Aggregator 5.1", layout="wide")

st.title("⚡ Energy Hedge & Profile Aggregator 5.1")
st.markdown("""
**Stap 1:** Upload ruwe meetdata (individuele aansluitingen). De app categoriseert en aggregeert deze automatisch.
**Stap 2:** Bepaal je inkoopstrategie op de berekende groepsprofielen.
""")

# --- DEEL 1: LOGICA VOOR CATEGORISEREN & AGGREGEREN ---

def calculate_winter_profile(df):
    # Wintermaanden: Jan (1), Nov (11), Dec (12)
    winter_mask = df.index.month.isin([1, 11, 12])
    winter_df = df[winter_mask]
    
    # Fallback: als er geen winterdata is, gebruik hele dataset
    if winter_df.empty:
        return df.groupby(df.index.time).mean()
    
    # FIX (v5.0): Gebruik de index van de subset om Grouper error te voorkomen
    return winter_df.groupby(winter_df.index.time).mean()

def estimate_gross_solar_robust(df, connection_col, winter_profile):
    col_data = df[connection_col]
    w_prof = winter_profile[connection_col]
    
    # Nachtverbruik bepalen
    night_mask = (df.index.hour < 6) | (df.index.hour >= 23)
    daily_night_avg = df.loc[night_mask, connection_col].resample('D').mean()
    
    # Basis nachtverbruik
    night_times = [t for t in w_prof.index if t.hour < 6 or t.hour >= 23]
    base_night_avg = w_prof.loc[night_times].mean()
    if base_night_avg < 0.05: base_night_avg = 0.05
    
    # Scaling
    daily_scaling = daily_night_avg / base_night_avg
    daily_scaling = daily_scaling.clip(0.2, 5.0)
    
    # Uitrollen naar kwartieren
    dates = pd.Series(df.index.normalize(), index=df.index)
    scaling_series = dates.map(daily_scaling).ffill().bfill()
    
    # Base Load
    base_load_series = df.index.map(lambda x: w_prof.loc[x.time()]) 
    base_load_series = pd.Series(base_load_series, index=df.index)
    expected_load = base_load_series * scaling_series
    
    # Solar calc
    solar_behind_meter = expected_load - col_data
    is_daylight = (df.index.hour >= 8) & (df.index.hour <= 20)
    solar_behind_meter = solar_behind_meter.where(is_daylight, 0).clip(lower=0)
    
    actual_export = col_data.clip(upper=0).abs()
    return solar_behind_meter + actual_export

@st.cache_data
def process_raw_connections(file):
    # Inlezen
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
    
    # Progress
    my_bar = st.progress(0, text="Analyseren aansluitingen...")
    total_cols = len(connection_cols)
    
    # Bruto Productie
    for i, col in enumerate(connection_cols):
        gross_series = estimate_gross_solar_robust(df, col, winter_profile)
        gross_prod_dict[col] = gross_series
        estimated_volumes[col] = gross_series.sum()
        if i % max(1, int(total_cols/10)) == 0:
            my_bar.progress((i + 1) / total_cols, text=f"Analyseren: {col}")
    my_bar.empty()
    
    gross_production_df = pd.DataFrame(gross_prod_dict, index=df.index)
    
    # Categorisatie 1
    categories = {}
    for col in connection_cols:
        gross_vol = estimated_volumes[col]
        if gross_vol > 1000: 
            total_import = df[col][df[col] > 0].sum()
            if total_import < 0.2 * gross_vol:
                categories[col] = 'Producer'
            else:
                categories[col] = 'Prosumer'
        else:
            categories[col] = 'Consumer'
            
    # Verfijning
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
            
    # Aggregatie
    cat_consumer = [c for c, cat in final_mapping.items() if cat == 'Consumer']
    cat_prosumer = [c for c, cat in final_mapping.items() if cat == 'Prosumer']
    cat_producer = [c for c, cat in final_mapping.items() if cat == 'Producer']
    
    agg_df = pd.DataFrame(index=df.index)
    agg_df['Consumer'] = df[cat_consumer].sum(axis=1) if cat_consumer else 0.0
    agg_df['Prosumer'] = df[cat_prosumer].sum(axis=1) if cat_prosumer else 0.0
    agg_df['Producer'] = df[cat_producer].sum(axis=1) if cat_producer else 0.0
    
    return agg_df, final_mapping

# --- DEEL 2: HEDGE OPTIMIZER ---

# Sidebar
st.sidebar.header("1. Data Input")
input_mode = st.sidebar.radio("Input Type", ["Ruwe Aansluitingen (CSV)", "Reeds Geaggregeerd (CSV)"])
uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

df_hedge = None

if uploaded_file is not None:
    if input_mode == "Ruwe Aansluitingen (CSV)":
        try:
            df_agg, mapping = process_raw_connections(uploaded_file)
            
            # Stats
            counts = pd.Series(mapping.values()).value_counts()
            c1, c2, c3 = st.columns(3)
            c1.metric("Consumers", counts.get('Consumer', 0))
            c2.metric("Prosumers", counts.get('Prosumer', 0))
            c3.metric("Producers", counts.get('Producer', 0))
            
            # FIX v5.1: Robuuste rename van index naar 'Date'
            df_hedge = df_agg.reset_index()
            # Hernoem de eerste kolom (de oude index) hard naar 'Date'
            cols = list(df_hedge.columns)
            cols[0] = 'Date'
            df_hedge.columns = cols

        except Exception as e:
            st.error(f"Fout bij verwerken ruwe data: {e}")
            st.stop()
            
    else: # Reeds geaggregeerd
        try:
            df_hedge = pd.read_csv(uploaded_file, sep=';', decimal=',')
            
            # FIX v5.1: Check of 'Date' bestaat, zo niet, zoek alternatieven
            if 'Date' not in df_hedge.columns:
                candidates = ['Datum', 'Tijd', 'Time', 'date', 'time']
                renamed = False
                for c in candidates:
                    if c in df_hedge.columns:
                        df_hedge.rename(columns={c: 'Date'}, inplace=True)
                        renamed = True
                        break
                if not renamed:
                    # Fallback: neem de eerste kolom als datum
                    cols = list(df_hedge.columns)
                    cols[0] = 'Date'
                    df_hedge.columns = cols

            # Cleaning numeriek
            for col in ['Consumer', 'Prosumer', 'Producer']:
                if col in df_hedge.columns:
                    df_hedge[col] = pd.to_numeric(df_hedge[col].astype(str).str.replace(',', '.'), errors='coerce')
            
            df_hedge['Date'] = pd.to_datetime(df_hedge['Date'], dayfirst=True)
            
        except Exception as e:
            st.error(f"Fout bij inlezen geaggregeerd bestand: {e}")
            st.stop()

# Hedge Logic
if df_hedge is not None:
    # Voorbereiding Data
    df = df_hedge.copy()
    
    # FIX v5.1: Zeker weten dat 'Date' datetime is voor sorteren
    if not pd.api.types.is_datetime64_any_dtype(df['Date']):
        df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)

    df = df.sort_values('Date')
    df = df.drop_duplicates(subset='Date', keep='first')
    df = df.set_index('Date').asfreq('15min').ffill().reset_index()

    # MW Calc
    for col in ['Consumer', 'Prosumer', 'Producer']:
        if col in df.columns:
            df[f'{col}_MW'] = (df[col] * 4) / 1000
        else:
            df[f'{col}_MW'] = 0.0
    
    df['is_peak'] = (df['Date'].dt.weekday < 5) & (df['Date'].dt.hour >= 8) & (df['Date'].dt.hour < 20)
    df['Quarter'] = df['Date'].dt.quarter

    # --- HEDGE CONFIG ---
    st.sidebar.markdown("---")
    st.sidebar.header("2. Hedge Configuratie")
    
    profile_choice = st.sidebar.selectbox("Kies Profiel", ["Consumer", "Prosumer", "Producer"])
    strategy_period = st.sidebar.radio("Periode", ["Per Jaar", "Per Kwartaal"])
    p_mw_col = f'{profile_choice}_MW'

    if 'slider_values' not in st.session_state: st.session_state['slider_values'] = {}

    # --- OPTIMIZER FUNCTIES ---
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
    st.sidebar.markdown("---")
    st.sidebar.header("3. Kies Strategie")

    def apply_strategy(strat_name):
        periods = [0] if strategy_period == "Per Jaar" else [1, 2, 3, 4]
        for q in periods:
            sub_df = df if q == 0 else df[df['Quarter'] == q]
            
            if strat_name == "0%_sell": b, p = find_optimal_mw(sub_df, target_over_pct_limit=0.1)
            elif strat_name == "5%_sell": b, p = find_optimal_mw(sub_df, target_over_pct_limit=5.0)
            elif strat_name == "80%_cov": b, p = find_optimal_mw(sub_df, percent_volume_target=80)
            elif strat_name == "100%_cov": b, p = find_optimal_mw(sub_df, percent_volume_target=100)
            elif strat_name == "110%_cov": b, p = find_optimal_mw(sub_df, percent_volume_target=110)
            
            key_b = f'slider_b_yr' if q == 0 else f'slider_b_q{q}'
            key_p = f'slider_p_yr' if q == 0 else f'slider_p_q{q}'
            st.session_state[key_b] = float(b)
            st.session_state[key_p] = float(p)

    c1, c2 = st.sidebar.columns(2)
    if c1.button("🛡️ 0% Sell"): apply_strategy("0%_sell")
    if c2.button("🎯 Max 5% Sell"): apply_strategy("5%_sell")
    c3, c4 = st.sidebar.columns(2)
    if c3.button("📉 Basis 80%"): apply_strategy("80%_cov")
    if c4.button("⚖️ Balans 100%"): apply_strategy("100%_cov")
    if st.sidebar.button("📈 Over-Hedge 110%", use_container_width=True): apply_strategy("110%_cov")

    # --- SLIDERS ---
    st.sidebar.markdown("---")
    st.sidebar.subheader("4. Fine-tuning (MW)")
    
    df['Current_Hedge_MW'] = 0.0
    slider_min, slider_max = -10.0, 25.0 

    if strategy_period == "Per Jaar":
        if 'slider_b_yr' not in st.session_state:
            def_b, def_p = find_optimal_mw(df, percent_volume_target=100)
            st.session_state['slider_b_yr'] = float(def_b)
            st.session_state['slider_p_yr'] = float(def_p)
            
        b_yr = st.sidebar.slider("Base (Jaar)", slider_min, slider_max, key="slider_b_yr", step=0.1)
        p_yr = st.sidebar.slider("Peak (Jaar)", slider_min, slider_max, key="slider_p_yr", step=0.1)
        df['Current_Hedge_MW'] = b_yr + (df['is_peak'] * p_yr)
    else:
        for q in [1, 2, 3, 4]:
            st.sidebar.markdown(f"**Kwartaal {q}**")
            q_mask = df['Quarter'] == q
            if f'slider_b_q{q}' not in st.session_state:
                def_b, def_p = find_optimal_mw(df[q_mask], percent_volume_target=100)
                st.session_state[f'slider_b_q{q}'] = float(def_b)
                st.session_state[f'slider_p_q{q}'] = float(def_p)
            
            c1, c2 = st.sidebar.columns(2)
            b_q = c1.slider(f"Q{q} Base", slider_min, slider_max, key=f"slider_b_q{q}", step=0.1)
            p_q = c2.slider(f"Q{q} Peak", slider_min, slider_max, key=f"slider_p_q{q}", step=0.1)
            df.loc[q_mask, 'Current_Hedge_MW'] = b_q + (df.loc[q_mask, 'is_peak'] * p_q)

    # --- RESULTATEN DASHBOARD ---
    df['Profile_MWh'] = df[p_mw_col] * 0.25
    df['Hedge_MWh'] = df['Current_Hedge_MW'] * 0.25
    df['Used_Hedge_MWh'] = np.minimum(df['Hedge_MWh'], df['Profile_MWh']) 
    df['Over_Hedge_MWh'] = np.maximum(0, df['Hedge_MWh'] - df['Profile_MWh'])
    df['Under_Hedge_MWh'] = np.maximum(0, df['Profile_MWh'] - df['Hedge_MWh'])

    total_prof = df['Profile_MWh'].sum()
    total_over = df['Over_Hedge_MWh'].sum()
    total_under = df['Under_Hedge_MWh'].sum()
    pct_over = (total_over / total_prof) * 100 if total_prof != 0 else 0
    pct_under = (total_under / total_prof) * 100 if total_prof != 0 else 0

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Effectieve Dekking", f"{(df['Used_Hedge_MWh'].sum()/total_prof)*100:.1f}%")
    k2.metric("Totaal Verbruik", f"{total_prof:,.0f} MWh")
    k3.metric("Teveel (Sell)", f"{total_over:,.0f} MWh", f"{pct_over:.1f}%", delta_color="inverse")
    k4.metric("Tekort (Buy)", f"{total_under:,.0f} MWh", f"{pct_under:.1f}%", delta_color="inverse")

    st.markdown("---")
    st.subheader("🔎 Seizoensanalyse")
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
                st.info("Geen data in deze periode")
                continue
            mask = (df['Date'] >= week['start']) & (df['Date'] <= pd.Timestamp(week['end']) + pd.Timedelta(days=1))
            chart_data = df.loc[mask].melt(id_vars=['Date'], value_vars=[p_mw_col, 'Current_Hedge_MW'], var_name='Type', value_name='MW')
            chart_data['Type'] = chart_data['Type'].replace({p_mw_col: 'Verbruik', 'Current_Hedge_MW': 'Hedge'})
            c = alt.Chart(chart_data).mark_line(interpolate='step-after').encode(
                x=alt.X('Date:T', axis=alt.Axis(format='%a %H:%M', title=None)),
                y=alt.Y('MW:Q', title=None), color=alt.Color('Type:N', legend=alt.Legend(orient='bottom', title=None))
            ).properties(height=180)
            st.altair_chart(c, use_container_width=True)

    csv_dl = df.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Download Resultaten (CSV)", csv_dl, "hedge_resultaten.csv", "text/csv")
else:
    st.info("👆 Upload een bestand om te beginnen.")
