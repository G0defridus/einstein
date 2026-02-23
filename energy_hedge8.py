import sys
import subprocess
import os

# --- AUTO-INSTALLER ---
# Dit blok controleert of alle benodigde packages geïnstalleerd zijn.
# Zo niet, dan installeert hij ze automatisch via pip op de achtergrond.
REQUIRED_PACKAGES = {
    "streamlit": "streamlit",
    "pandas": "pandas",
    "numpy": "numpy",
    "altair": "altair",
    "entsoe": "entsoe-py"
}

for module_name, pip_name in REQUIRED_PACKAGES.items():
    try:
        __import__(module_name)
    except ImportError:
        print(f"⏳ Module '{module_name}' ontbreekt. Bezig met automatisch installeren van '{pip_name}'...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", pip_name])
        print(f"✅ '{pip_name}' succesvol geïnstalleerd!")

# Nu we zeker weten dat alles er is, kunnen we veilig importeren:
import streamlit as st
import pandas as pd
import numpy as np
import altair as alt

# Pagina instellingen
st.set_page_config(page_title="Energy Hedge Optimizer 8.1 (incl. EPEX)", layout="wide")

st.title("⚡ Energy Hedge Optimizer 8.1")

# --- DOCUMENTATIE BLOK (Inklapbaar) ---
with st.expander("📘 Lees mij: Achtergrond en Methodiek (Klik om te openen)", expanded=False):
    st.markdown("""
    ### 1. Van Ruwe Data naar Profiel
    Het model analyseert slimme meter data om te bepalen of een aansluiting een **Consumer**, **Prosumer** of **Producer** is.
    * **Stap 1 (Winterprofiel):** We kijken naar Jan/Nov/Dec om het basisverbruik te bepalen.
    * **Stap 2 (Zon Detectie):** We vergelijken de zomer met de winter. Het verschil is de bruto zonne-opwek.
    * **Stap 3 (Beslisboom):** Op basis van ratio's wordt de aansluiting toegekend aan een profiel.
    
    ### 2. Hedge Strategie
    We kopen in op de groothandelsmarkt in blokken van **0,1 MW**.
    * **Base:** 24/7 vast vermogen.
    * **Peak:** Extra vermogen op werkdagen (08:00 - 20:00).
    * **Total:** Inkoop voor het samengestelde portfolio-profiel (salderen).
    
    ### 3. Financiële Waardering (EPEX)
    Via de ENTSO-E API wordt de **Day-Ahead Spotprijs (EPEX)** opgehaald.
    * **Spot Inkoop (Kosten):** Volume dat je tekort komt (Under-hedge) × EPEX prijs.
    * **Spot Verkoop (Opbrengst):** Volume dat je over hebt (Over-hedge) × EPEX prijs. Let op: bij negatieve prijzen kost verkopen geld!
    * **Netto Spot Resultaat:** Verkoopopbrengst - Inkoopkosten. (Positief = je verdient aan de spotmarkt, Negatief = je betaalt bij op de spotmarkt).
    """)

# --- DEEL 1: LOGICA ---
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

# --- DEEL 2: HEDGE OPTIMIZER UI ---
st.sidebar.header("1. Data Input")
input_mode = st.sidebar.radio("Input Type", ["Ruwe Aansluitingen (CSV)", "Reeds Geaggregeerd (CSV)"])
uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

df_hedge = None

if uploaded_file is not None:
    if input_mode == "Ruwe Aansluitingen (CSV)":
        try:
            df_agg, mapping = process_raw_connections(uploaded_file)
            with st.expander("ℹ️ Resultaat Analyse & Categorisatie (Klik voor details)", expanded=True):
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

# Hedge Logic
if df_hedge is not None:
    df = df_hedge.copy()
    if not pd.api.types.is_datetime64_any_dtype(df['Date']):
        df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)

    df = df.sort_values('Date').drop_duplicates(subset='Date', keep='first')
    df = df.set_index('Date').asfreq('15min').ffill().reset_index()

    for col in ['Consumer', 'Prosumer
