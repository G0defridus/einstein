import streamlit as st
import pandas as pd
import datetime

# Pagina instellingen
st.set_page_config(page_title="Energy Hedge Simulator Pro", layout="wide")

st.title("⚡ Energy Hedge Simulator: Seizoensanalyse")
st.markdown("""
Stel je inkoopblokken in (stapgrootte 0,1 MW) en bekijk direct de impact in 4 representatieve weken door het jaar heen.
""")

# --- AANGEPAST: Data via File Uploader ---
st.sidebar.header("1. Data Input")
uploaded_file = st.sidebar.file_uploader("Sleep je CSV bestand hierheen", type=["csv"])

@st.cache_data
def load_data(file):
    # Let op: we lezen nu het 'file' object in plaats van een bestandsnaam
    df = pd.read_csv(file, sep=';', decimal=',')
    
    # Numeriek maken en opschonen (hetzelfde als eerst)
    for col in ['Consumer', 'Prosumer', 'Producer']:
        df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', '.'), errors='coerce')
    
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
    df = df.dropna()
    
    # Omrekenen naar MW
    for col in ['Consumer', 'Prosumer', 'Producer']:
        df[f'{col}_MW'] = (df[col] * 4) / 1000
        
    df['is_peak'] = (df['Date'].dt.weekday < 5) & (df['Date'].dt.hour >= 8) & (df['Date'].dt.hour < 20)
    df['Quarter'] = df['Date'].dt.quarter
    return df

if uploaded_file is not None:
    try:
        df = load_data(uploaded_file)
        # HIERONDER KOMT DE REST VAN JE CODE (vanaf 'st.sidebar.header("2. Configuratie")'...)
    except Exception as e:
        st.error(f"Fout bij inlezen bestand: {e}")
        st.stop()
else:
    st.info("👆 Upload eerst een CSV-bestand in de zijbalk om te beginnen.")
    st.stop() # Stopt de app totdat er een bestand is

# --- 2. Sidebar Instellingen ---
st.sidebar.header("1. Configuratie")
profile_choice = st.sidebar.selectbox("Kies Profiel", ["Consumer", "Prosumer", "Producer"])
strategy = st.sidebar.radio("Strategie Periode", ["Per Jaar", "Per Kwartaal"])

st.sidebar.markdown("---")
st.sidebar.header("2. Instelling Hedge")
mode = st.sidebar.radio("Wijze van instellen", ["Via Percentage (%)", "Handmatig (MW)"])

p_mw_col = f'{profile_choice}_MW'
df['Current_Hedge_MW'] = 0.0

# Hulpvariabelen voor slider ranges (dynamisch op basis van profiel max/min)
p_max = df[p_mw_col].max()
p_min = df[p_mw_col].min()
# Ruime marges voor de sliders
slider_min = float(min(-5, p_min * 1.5))
slider_max = float(max(5, p_max * 1.5))

# Functie om automatische MWs te berekenen op basis van een dataset slice en percentage
def calc_auto_mw(sub_df, percentage):
    # Logica: Base = Gemiddelde tijdens Off-Peak uren * %
    # Peak Add-on = (Gemiddelde tijdens Peak uren - Gemiddelde tijdens Off-Peak uren) * %
    
    # Filter off-peak en peak
    off_peak_mean = sub_df.loc[~sub_df['is_peak'], p_mw_col].mean()
    peak_mean = sub_df.loc[sub_df['is_peak'], p_mw_col].mean()
    
    # Bereken targets
    target_base = off_peak_mean * (percentage / 100.0)
    target_peak_total = peak_mean * (percentage / 100.0)
    target_peak_addon = target_peak_total - target_base
    
    # Afronden op 0.1 MW
    return round(target_base, 1), round(target_peak_addon, 1)

# --- LOGICA VOOR SLIDERS EN BEREKENING ---

if mode == "Via Percentage (%)":
    # Eén hoofdschuif voor het percentage
    pct = st.sidebar.slider("Hedge Doelstelling (%)", 0, 200, 100, 5)
    
    if strategy == "Per Jaar":
        # Bereken jaarwaarden
        b_yr, p_yr = calc_auto_mw(df, pct)
        st.sidebar.info(f"**Berekende Blokken (Jaar):**\n\nBase: {b_yr} MW\nPeak: {p_yr} MW")
        df['Current_Hedge_MW'] = b_yr + (df['is_peak'] * p_yr)
        
    else: # Per Kwartaal
        st.sidebar.markdown("**Berekende Blokken per Kwartaal:**")
        for q in [1, 2, 3, 4]:
            q_mask = df['Quarter'] == q
            b_q, p_q = calc_auto_mw(df.loc[q_mask], pct)
            st.sidebar.text(f"Q{q}: Base {b_q} | Peak {p_q}")
            df.loc[q_mask, 'Current_Hedge_MW'] = b_q + (df.loc[q_mask, 'is_peak'] * p_q)

else: # Handmatig (MW)
    st.sidebar.markdown("*Stel de MW waarden handmatig in met de sliders.*")
    
    if strategy == "Per Jaar":
        # Defaults (gemiddelden)
        avg_base, avg_peak_add = calc_auto_mw(df, 100)
        
        b_yr = st.sidebar.slider("Base MW (Jaar)", slider_min, slider_max, float(avg_base), 0.1)
        p_yr = st.sidebar.slider("Peak Extra MW (Jaar)", slider_min, slider_max, float(avg_peak_add), 0.1)
        
        df['Current_Hedge_MW'] = b_yr + (df['is_peak'] * p_yr)
        
    else: # Per Kwartaal
        for q in [1, 2, 3, 4]:
            st.sidebar.markdown(f"**Kwartaal {q}**")
            q_mask = df['Quarter'] == q
            # Defaults
            def_b, def_p = calc_auto_mw(df.loc[q_mask], 100)
            
            c1, c2 = st.sidebar.columns(2)
            with c1:
                b_q = st.slider(f"Q{q} Base", slider_min, slider_max, float(def_b), 0.1, key=f"b_{q}")
            with c2:
                p_q = st.slider(f"Q{q} Peak", slider_min, slider_max, float(def_p), 0.1, key=f"p_{q}")
            
            df.loc[q_mask, 'Current_Hedge_MW'] = b_q + (df.loc[q_mask, 'is_peak'] * p_q)

# --- 3. Resultaten & KPI's ---
df['Profile_MWh'] = df[p_mw_col] * 0.25
df['Hedge_MWh'] = df['Current_Hedge_MW'] * 0.25
df['Diff_MWh'] = df['Hedge_MWh'] - df['Profile_MWh']

total_prof = df['Profile_MWh'].sum()
total_hedge = df['Hedge_MWh'].sum()
over_mwh = df.loc[df['Diff_MWh'] > 0, 'Diff_MWh'].sum()
under_mwh = df.loc[df['Diff_MWh'] < 0, 'Diff_MWh'].abs().sum()

# KPI Display
kpi1, kpi2, kpi3, kpi4 = st.columns(4)
kpi1.metric("Hedge Dekking", f"{(total_hedge/total_prof)*100:.1f}%")
kpi2.metric("Totaal Verbruik", f"{total_prof:,.0f} MWh")
kpi3.metric("Teveel (Sell)", f"{over_mwh:,.0f} MWh", delta_color="inverse")
kpi4.metric("Tekort (Buy)", f"{under_mwh:,.0f} MWh", delta_color="inverse")

# --- 4. Visuele Weergave (Verbeterd met Altair Steps) ---
st.markdown("---")
st.subheader("🔎 Seizoensanalyse (4 weken)")
import altair as alt

weeks = [
    {"name": "Februari (Winter)", "start": "2025-02-03", "end": "2025-02-09"},
    {"name": "Mei (Lente)",       "start": "2025-05-05", "end": "2025-05-11"},
    {"name": "Augustus (Zomer)",  "start": "2025-08-04", "end": "2025-08-10"},
    {"name": "November (Herfst)", "start": "2025-11-03", "end": "2025-11-09"}
]

r1_cols = st.columns(2)
r2_cols = st.columns(2)
all_cols = r1_cols + r2_cols

for i, week in enumerate(weeks):
    with all_cols[i]:
        st.markdown(f"**{week['name']}**")
        
        # Data filteren
        mask = (df['Date'] >= week['start']) & (df['Date'] <= pd.Timestamp(week['end']) + pd.Timedelta(days=1))
        chart_data = df.loc[mask].copy()
        
        if not chart_data.empty:
            # Data omvormen voor Altair (Long format is makkelijker voor legenda's)
            chart_melt = chart_data.melt(id_vars=['Date'], value_vars=[p_mw_col, 'Current_Hedge_MW'], 
                                         var_name='Type', value_name='MW')
            
            # De namen wat mooier maken voor de legenda
            chart_melt['Type'] = chart_melt['Type'].replace({p_mw_col: 'Verbruik', 'Current_Hedge_MW': 'Hedge Blok'})

            # De Grafiek
            c = alt.Chart(chart_melt).mark_line(
                interpolate='step-after',  # DIT ZORGT VOOR DE RECHTE BLOKKEN
                strokeWidth=2
            ).encode(
                x=alt.X('Date:T', axis=alt.Axis(format='%a %H:%M', title='')),
                y=alt.Y('MW:Q', title='Vermogen (MW)'),
                color=alt.Color('Type:N', legend=alt.Legend(title=None, orient='bottom')),
                tooltip=['Date', 'Type', 'MW']
            ).properties(
                height=250
            )
            
            st.altair_chart(c, use_container_width=True)

# --- 5. Kwartaal Tabel (Terug van weggeweest) ---
st.markdown("---")
st.subheader("📊 Resultaten per Kwartaal")

q_stats = df.groupby('Quarter').apply(lambda x: pd.Series({
    'Verbruik (MWh)': x['Profile_MWh'].sum(),
    'Hedge (MWh)': x['Hedge_MWh'].sum(),
    'Dekking %': (x['Hedge_MWh'].sum() / x['Profile_MWh'].sum() * 100) if x['Profile_MWh'].sum() != 0 else 0,
    'Long/Verkoop (MWh)': x.loc[x['Diff_MWh'] > 0, 'Diff_MWh'].sum(),
    'Short/Inkoop (MWh)': x.loc[x['Diff_MWh'] < 0, 'Diff_MWh'].abs().sum()
}))

# Mooi formatteren
st.dataframe(q_stats.style.format({
    'Verbruik (MWh)': "{:,.0f}",
    'Hedge (MWh)': "{:,.0f}",
    'Dekking %': "{:.1f}%",
    'Long/Verkoop (MWh)': "{:,.0f}",
    'Short/Inkoop (MWh)': "{:,.0f}"
}), use_container_width=True)

# --- 6. Export ---
csv_dl = df.to_csv(index=False).encode('utf-8')
st.download_button("📥 Download Data (CSV)", csv_dl, "hedge_scenario_v2.csv", "text/csv")