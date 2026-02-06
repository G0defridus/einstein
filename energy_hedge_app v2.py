import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import os

# Pagina instellingen
st.set_page_config(page_title="Energy Hedge Optimizer 4.2", layout="wide")

st.title("⚡ Energy Hedge Optimizer 4.2")
st.markdown("""
Kies een automatische strategie of stel de blokken handmatig bij. 
De optimizer berekent de ideale MW-waarden op basis van je gekozen risicoprofiel.
""")

# --- 1. Data Laden ---
st.sidebar.header("1. Data Input")
uploaded_file = st.sidebar.file_uploader("Upload CSV (optioneel)", type=["csv"])

@st.cache_data
def load_data(file):
    if file is None:
        if not os.path.exists('drie_categorien.csv'): return None
        file = 'drie_categorien.csv'
    
    df = pd.read_csv(file, sep=';', decimal=',')
    
    # Cleaning
    for col in ['Consumer', 'Prosumer', 'Producer']:
        df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', '.'), errors='coerce')
    
    # Datum fix
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
    df = df.sort_values('Date')
    df = df.drop_duplicates(subset='Date', keep='first') # Wintertijd fix
    
    # Resample
    df = df.set_index('Date').asfreq('15min').ffill().reset_index()

    # MW Calc
    for col in ['Consumer', 'Prosumer', 'Producer']:
        df[f'{col}_MW'] = (df[col] * 4) / 1000
    
    # Features
    df['is_peak'] = (df['Date'].dt.weekday < 5) & (df['Date'].dt.hour >= 8) & (df['Date'].dt.hour < 20)
    df['Quarter'] = df['Date'].dt.quarter
    return df

try:
    df = load_data(uploaded_file)
    if df is None:
        st.warning("⚠️ Upload een bestand om te beginnen.")
        st.stop()
except Exception as e:
    st.error(f"Error: {e}")
    st.stop()

# --- 2. Profiel & Periode ---
st.sidebar.header("2. Configuratie")
profile_choice = st.sidebar.selectbox("Kies Profiel", ["Consumer", "Prosumer", "Producer"])
strategy_period = st.sidebar.radio("Periode", ["Per Jaar", "Per Kwartaal"])
p_mw_col = f'{profile_choice}_MW'

# Bepaal dynamische slider ranges op basis van de data
p_max = df[p_mw_col].max()
p_min = df[p_mw_col].min()
# Zorg voor marge zodat sliders niet vastlopen
slider_min = float(min(-5, p_min * 1.5))
slider_max = float(max(10, p_max * 1.5))

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
    # 1. Simpele volume targets
    if percent_volume_target is not None:
        off_peak_mean = sub_df.loc[~sub_df['is_peak'], p_mw_col].mean()
        peak_mean = sub_df.loc[sub_df['is_peak'], p_mw_col].mean()
        if pd.isna(off_peak_mean): off_peak_mean = 0.0
        if pd.isna(peak_mean): peak_mean = 0.0
        
        b = off_peak_mean * (percent_volume_target / 100.0)
        p_add = (peak_mean * (percent_volume_target / 100.0)) - b
        return round(b, 2), round(p_add, 2)

    # 2. Complexe limiet targets (Iteratief)
    best_b, best_p = 0.0, 0.0
    # Scan van hoog naar laag
    for pct in range(150, 0, -2): 
        b_try, p_try = find_optimal_mw(sub_df, percent_volume_target=pct)
        _, _, over_pct = calculate_metrics(sub_df, b_try, p_try)
        
        if target_over_pct_limit is not None and over_pct <= target_over_pct_limit:
            best_b, best_p = b_try, p_try
            break 
            
    return best_b, best_p

# --- 3. Strategie Knoppen ---
st.sidebar.markdown("---")
st.sidebar.header("3. Kies een Strategie")

def apply_strategy(strat_name):
    periods = [0] if strategy_period == "Per Jaar" else [1, 2, 3, 4]
    
    for q in periods:
        if q == 0: sub_df = df
        else: sub_df = df[df['Quarter'] == q]
        
        # Bereken waarden
        if strat_name == "0%_sell":
            b, p = find_optimal_mw(sub_df, target_over_pct_limit=0.1)
        elif strat_name == "5%_sell":
            b, p = find_optimal_mw(sub_df, target_over_pct_limit=5.0)
        elif strat_name == "80%_cov":
            b, p = find_optimal_mw(sub_df, percent_volume_target=80)
        elif strat_name == "100%_cov":
            b, p = find_optimal_mw(sub_df, percent_volume_target=100)
        elif strat_name == "110%_cov":
            b, p = find_optimal_mw(sub_df, percent_volume_target=110)
        
        # FIX: Update direct de Session State Keys van de sliders
        if q == 0: # Jaar
            st.session_state['slider_b_yr'] = float(b)
            st.session_state['slider_p_yr'] = float(p)
        else: # Kwartaal
            st.session_state[f'slider_b_q{q}'] = float(b)
            st.session_state[f'slider_p_q{q}'] = float(p)

# De Knoppen
c1, c2 = st.sidebar.columns(2)
if c1.button("🛡️ 0% Sell", help="Inkoop verlagen tot verkoop nagenoeg 0 is"): apply_strategy("0%_sell")
if c2.button("🎯 Max 5% Sell", help="Max hedge zolang verkoop < 5%"): apply_strategy("5%_sell")

c3, c4 = st.sidebar.columns(2)
if c3.button("📉 Basis 80%", help="Veilige ondergrens"): apply_strategy("80%_cov")
if c4.button("⚖️ Balans 100%", help="Gemiddeld verbruik"): apply_strategy("100%_cov")

if st.sidebar.button("📈 Over-Hedge 110%", use_container_width=True): apply_strategy("110%_cov")

# --- 4. Sliders ---
st.sidebar.markdown("---")
st.sidebar.subheader("4. Fine-tuning (MW)")

df['Current_Hedge_MW'] = 0.0

if strategy_period == "Per Jaar":
    # Defaults alleen voor eerste keer laden
    def_b, def_p = find_optimal_mw(df, percent_volume_target=100)
    
    # Gebruik key zodat we deze vanuit de knoppen kunnen overschrijven
    b_yr = st.sidebar.slider("Base (Jaar)", slider_min, slider_max, float(def_b), 0.1, key="slider_b_yr")
    p_yr = st.sidebar.slider("Peak (Jaar)", slider_min, slider_max, float(def_p), 0.1, key="slider_p_yr")
    
    df['Current_Hedge_MW'] = b_yr + (df['is_peak'] * p_yr)

else: # Kwartaal
    for q in [1, 2, 3, 4]:
        st.sidebar.markdown(f"**Kwartaal {q}**")
        q_mask = df['Quarter'] == q
        def_b, def_p = find_optimal_mw(df[q_mask], percent_volume_target=100)
        
        c1, c2 = st.sidebar.columns(2)
        b_q = c1.slider(f"Q{q} Base", slider_min, slider_max, float(def_b), 0.1, key=f"slider_b_q{q}")
        p_q = c2.slider(f"Q{q} Peak", slider_min, slider_max, float(def_p), 0.1, key=f"slider_p_q{q}")
        
        df.loc[q_mask, 'Current_Hedge_MW'] = b_q + (df.loc[q_mask, 'is_peak'] * p_q)

# --- 5. Resultaten ---
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

# KPI's
k1, k2, k3, k4 = st.columns(4)
k1.metric("Effectieve Dekking", f"{(df['Used_Hedge_MWh'].sum()/total_prof)*100:.1f}%")
k2.metric("Totaal Verbruik", f"{total_prof:,.0f} MWh")
k3.metric("Teveel (Sell)", f"{total_over:,.0f} MWh", f"{pct_over:.1f}%", delta_color="inverse")
k4.metric("Tekort (Buy)", f"{total_under:,.0f} MWh", f"{pct_under:.1f}%", delta_color="inverse")

# Grafieken
st.markdown("---")
weeks = [
    {"name": "Februari (Winter)", "start": "2025-02-03", "end": "2025-02-09"},
    {"name": "Mei (Lente)",       "start": "2025-05-05", "end": "2025-05-11"},
    {"name": "Augustus (Zomer)",  "start": "2025-08-04", "end": "2025-08-10"},
    {"name": "November (Herfst)", "start": "2025-11-03", "end": "2025-11-09"}
]
cols = st.columns(2) + st.columns(2)
for i, week in enumerate(weeks):
    with cols[i]:
        st.markdown(f"**{week['name']}**")
        mask = (df['Date'] >= week['start']) & (df['Date'] <= pd.Timestamp(week['end']) + pd.Timedelta(days=1))
        chart_data = df.loc[mask].melt(id_vars=['Date'], value_vars=[p_mw_col, 'Current_Hedge_MW'], var_name='Type', value_name='MW')
        chart_data['Type'] = chart_data['Type'].replace({p_mw_col: 'Verbruik', 'Current_Hedge_MW': 'Hedge'})
        
        c = alt.Chart(chart_data).mark_line(interpolate='step-after').encode(
            x=alt.X('Date:T', axis=alt.Axis(format='%a %H:%M', title=None)),
            y=alt.Y('MW:Q'), color=alt.Color('Type:N', legend=alt.Legend(orient='bottom', title=None))
        ).properties(height=200)
        st.altair_chart(c, use_container_width=True)

# Tabel
st.markdown("---")
q_stats = df.groupby('Quarter').apply(lambda x: pd.Series({
    'Verbruik': x['Profile_MWh'].sum(),
    'Dekking %': (x['Used_Hedge_MWh'].sum() / x['Profile_MWh'].sum() * 100) if x['Profile_MWh'].sum() != 0 else 0,
    'Teveel %': (x['Over_Hedge_MWh'].sum() / x['Profile_MWh'].sum() * 100) if x['Profile_MWh'].sum() != 0 else 0,
}))
st.dataframe(q_stats.style.format("{:.1f}"))