import streamlit as st
import pandas as pd
import numpy as np
import altair as alt

# Pagina instellingen
st.set_page_config(page_title="Energy Hedge Simulator 3.0", layout="wide")

st.title("⚡ Energy Hedge Simulator 3.0")
st.markdown("""
**Nieuwe definitie:** De 'Dekking' telt nu alleen de daadwerkelijk benutte energie (max 100%).
Het overschot wordt apart weergegeven als percentage dat je moet terugverkopen.
""")

# --- 1. Data Laden (Inclusief Wintertijd Fix) ---
@st.sidebar.header("1. Data Input")
uploaded_file = st.sidebar.file_uploader("Upload CSV (optioneel)", type=["csv"])

@st.cache_data
def load_data(file):
    # Als er geen file is geüpload, gebruik lokaal bestand (fallback)
    if file is None:
        file = 'drie_categorien.csv'
    
    df = pd.read_csv(file, sep=';', decimal=',')
    
    # Numeriek maken
    for col in ['Consumer', 'Prosumer', 'Producer']:
        df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', '.'), errors='coerce')
    
    # Datum parsen en sorteren
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
    df = df.sort_values('Date')
    
    # Dubbele indexen verwijderen (Wintertijd fix)
    df = df.drop_duplicates(subset='Date', keep='first')
    
    # Tijdlijn repareren (Gaten vullen)
    df = df.set_index('Date')
    df = df.asfreq('15min') 
    df = df.ffill()
    df = df.reset_index()

    # Omrekenen naar MW
    for col in ['Consumer', 'Prosumer', 'Producer']:
        df[f'{col}_MW'] = (df[col] * 4) / 1000
    
    # Tijdkenmerken
    df['is_peak'] = (df['Date'].dt.weekday < 5) & (df['Date'].dt.hour >= 8) & (df['Date'].dt.hour < 20)
    df['Quarter'] = df['Date'].dt.quarter
    
    return df

try:
    # Check of bestand bestaat (voor lokaal gebruik zonder upload)
    import os
    if uploaded_file is not None:
        df = load_data(uploaded_file)
    elif os.path.exists('drie_categorien.csv'):
        df = load_data(None)
    else:
        st.warning("Upload een CSV bestand om te beginnen.")
        st.stop()
except Exception as e:
    st.error(f"Fout bij laden: {e}")
    st.stop()

# --- 2. Sidebar Instellingen ---
st.sidebar.header("2. Configuratie")
profile_choice = st.sidebar.selectbox("Kies Profiel", ["Consumer", "Prosumer", "Producer"])
strategy = st.sidebar.radio("Strategie Periode", ["Per Jaar", "Per Kwartaal"])

st.sidebar.markdown("---")
st.sidebar.header("3. Instelling Hedge")
mode = st.sidebar.radio("Wijze van instellen", ["Via Percentage (%)", "Handmatig (MW)"])

p_mw_col = f'{profile_choice}_MW'
df['Current_Hedge_MW'] = 0.0

# Hulpvariabelen voor slider ranges
p_max = df[p_mw_col].max()
p_min = df[p_mw_col].min()
slider_min = float(min(-5, p_min * 1.5))
slider_max = float(max(5, p_max * 1.5))

def calc_auto_mw(sub_df, percentage):
    off_peak_mean = sub_df.loc[~sub_df['is_peak'], p_mw_col].mean()
    peak_mean = sub_df.loc[sub_df['is_peak'], p_mw_col].mean()
    target_base = off_peak_mean * (percentage / 100.0)
    target_peak_addon = (peak_mean * (percentage / 100.0)) - target_base
    return round(target_base, 1), round(target_peak_addon, 1)

# --- LOGICA SLIDERS ---
if mode == "Via Percentage (%)":
    pct = st.sidebar.slider("Hedge Doelstelling (%)", 0, 200, 100, 5)
    if strategy == "Per Jaar":
        b_yr, p_yr = calc_auto_mw(df, pct)
        st.sidebar.info(f"**Jaar:** Base {b_yr} | Peak {p_yr}")
        df['Current_Hedge_MW'] = b_yr + (df['is_peak'] * p_yr)
    else:
        st.sidebar.markdown("**Berekend per Kwartaal:**")
        for q in [1, 2, 3, 4]:
            q_mask = df['Quarter'] == q
            b_q, p_q = calc_auto_mw(df.loc[q_mask], pct)
            st.sidebar.caption(f"Q{q}: Base {b_q} | Peak {p_q}")
            df.loc[q_mask, 'Current_Hedge_MW'] = b_q + (df.loc[q_mask, 'is_peak'] * p_q)
else:
    if strategy == "Per Jaar":
        avg_base, avg_peak_add = calc_auto_mw(df, 100)
        b_yr = st.sidebar.slider("Base MW (Jaar)", slider_min, slider_max, float(avg_base), 0.1)
        p_yr = st.sidebar.slider("Peak Extra MW (Jaar)", slider_min, slider_max, float(avg_peak_add), 0.1)
        df['Current_Hedge_MW'] = b_yr + (df['is_peak'] * p_yr)
    else:
        for q in [1, 2, 3, 4]:
            st.sidebar.markdown(f"**Kwartaal {q}**")
            q_mask = df['Quarter'] == q
            def_b, def_p = calc_auto_mw(df.loc[q_mask], 100)
            c1, c2 = st.sidebar.columns(2)
            with c1: b_q = st.slider(f"Q{q} Base", slider_min, slider_max, float(def_b), 0.1, key=f"b_{q}")
            with c2: p_q = st.slider(f"Q{q} Peak", slider_min, slider_max, float(def_p), 0.1, key=f"p_{q}")
            df.loc[q_mask, 'Current_Hedge_MW'] = b_q + (df.loc[q_mask, 'is_peak'] * p_q)

# --- 3. Resultaten & KPI's (AANGEPAST) ---
df['Profile_MWh'] = df[p_mw_col] * 0.25
df['Hedge_MWh'] = df['Current_Hedge_MW'] * 0.25

# Verschil berekenen
df['Diff_MWh'] = df['Hedge_MWh'] - df['Profile_MWh']

# NIEUWE LOGICA:
# 1. Effectief Gebruikt = Het minimum van wat je hebt en wat je nodig hebt
df['Used_Hedge_MWh'] = np.minimum(df['Hedge_MWh'], df['Profile_MWh']) 
# (Let op: bij negatieve profielen zoals Producer werkt minimum anders, 
# maar voor inkoop (positief) is dit correct. Voor producer (negatief) is 'gebruikt' eigenlijk 'verkocht volume gedekt'.
# We nemen hier de absolute volumes voor de KPI's om het begrijpelijk te houden voor consument/prosumer)

# 2. Over-hedge (Teveel) = Alleen als Hedge > Profile
df['Over_Hedge_MWh'] = np.maximum(0, df['Hedge_MWh'] - df['Profile_MWh'])

# 3. Under-hedge (Tekort) = Alleen als Profile > Hedge
df['Under_Hedge_MWh'] = np.maximum(0, df['Profile_MWh'] - df['Hedge_MWh'])

# Totalen
total_prof = df['Profile_MWh'].sum()
total_used = df['Used_Hedge_MWh'].sum()
total_over = df['Over_Hedge_MWh'].sum()
total_under = df['Under_Hedge_MWh'].sum()

# Percentages t.o.v. Totaal Verbruik
pct_covered = (total_used / total_prof) * 100 if total_prof != 0 else 0
pct_over = (total_over / total_prof) * 100 if total_prof != 0 else 0
pct_under = (total_under / total_prof) * 100 if total_prof != 0 else 0

# KPI Display
kpi1, kpi2, kpi3, kpi4 = st.columns(4)
kpi1.metric("Effectieve Dekking", f"{pct_covered:.1f}%", help="Percentage van je verbruik dat gedekt is door de hedge.")
kpi2.metric("Totaal Verbruik", f"{total_prof:,.0f} MWh")
kpi3.metric("Teveel (Terugverkoop)", f"{total_over:,.0f} MWh", delta=f"{pct_over:.1f}% van verbruik", delta_color="inverse")
kpi4.metric("Tekort (Spot Inkoop)", f"{total_under:,.0f} MWh", delta=f"{pct_under:.1f}% van verbruik", delta_color="inverse")

# --- 4. Visuele Weergave (Altair Steps) ---
st.markdown("---")
st.subheader("🔎 Seizoensanalyse (4 weken)")

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
        mask = (df['Date'] >= week['start']) & (df['Date'] <= pd.Timestamp(week['end']) + pd.Timedelta(days=1))
        chart_data = df.loc[mask].copy()
        
        if not chart_data.empty:
            chart_melt = chart_data.melt(id_vars=['Date'], value_vars=[p_mw_col, 'Current_Hedge_MW'], var_name='Type', value_name='MW')
            chart_melt['Type'] = chart_melt['Type'].replace({p_mw_col: 'Verbruik', 'Current_Hedge_MW': 'Hedge Blok'})

            c = alt.Chart(chart_melt).mark_line(interpolate='step-after', strokeWidth=2).encode(
                x=alt.X('Date:T', axis=alt.Axis(format='%a %H:%M', title='')),
                y=alt.Y('MW:Q', title='MW'),
                color=alt.Color('Type:N', legend=alt.Legend(title=None, orient='bottom'))
            ).properties(height=200)
            st.altair_chart(c, use_container_width=True)

# --- 5. Tabel (Updated met nieuwe definities) ---
st.markdown("---")
st.subheader("📊 Kwartaal Balans")

q_stats = df.groupby('Quarter').apply(lambda x: pd.Series({
    'Verbruik (MWh)': x['Profile_MWh'].sum(),
    'Effectief Gedekt (MWh)': x['Used_Hedge_MWh'].sum(),
    'Dekking %': (x['Used_Hedge_MWh'].sum() / x['Profile_MWh'].sum() * 100) if x['Profile_MWh'].sum() != 0 else 0,
    'Teveel / Verkoop (MWh)': x['Over_Hedge_MWh'].sum(),
    'Over-hedge %': (x['Over_Hedge_MWh'].sum() / x['Profile_MWh'].sum() * 100) if x['Profile_MWh'].sum() != 0 else 0,
    'Tekort / Inkoop (MWh)': x['Under_Hedge_MWh'].sum()
}))

st.dataframe(q_stats.style.format({
    'Verbruik (MWh)': "{:,.0f}",
    'Effectief Gedekt (MWh)': "{:,.0f}",
    'Dekking %': "{:.1f}%",
    'Teveel / Verkoop (MWh)': "{:,.0f}",
    'Over-hedge %': "{:.1f}%",
    'Tekort / Inkoop (MWh)': "{:,.0f}"
}), use_container_width=True)

# Export
csv_dl = df.to_csv(index=False).encode('utf-8')
st.download_button("📥 Download Detail Data (CSV)", csv_dl, "hedge_resultaten_v3.csv", "text/csv")