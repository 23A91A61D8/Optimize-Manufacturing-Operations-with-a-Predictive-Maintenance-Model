import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from models.predictive_model import train_model, get_shap_values
import shap

# ---------------------------
# LOAD DATA (Sample data)
# ---------------------------
@st.cache_data(ttl=3600)
def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    np.random.seed(42)
    machine_ids = np.arange(1, 101)
    dates = pd.date_range(end=pd.Timestamp.today(), periods=100)

    data = []
    for m in machine_ids:
        risk_base = np.random.rand()
        for d in dates:
            tool_wear = np.random.normal(50, 10)
            temp = np.random.normal(70, 5)
            speed = np.random.normal(1500, 300)
            torque = np.random.normal(200, 40)
            failure = 1 if np.random.rand() < 0.02 else 0
            risk_score = min(1, risk_base + (tool_wear / 100) * 0.4 + (temp / 100) * 0.3 + (torque / 300) * 0.3)
            data.append([m, d, tool_wear, temp, speed, torque, failure, risk_score])

    df = pd.DataFrame(data, columns=[
        'machine_udi', 'timestamp', 'tool_wear', 'process_temperature',
        'rotational_speed', 'torque', 'failure_event', 'predicted_risk_score'
    ])

    # Factor importance (sample)
    factors = []
    for m in machine_ids:
        factors.append({
            'machine_udi': m,
            'Tool wear': np.round(np.random.rand(), 2),
            'Process temperature': np.round(np.random.rand(), 2),
            'Rotational speed': np.round(np.random.rand(), 2),
            'Torque': np.round(np.random.rand(), 2),
        })
    df_factors = pd.DataFrame(factors)

    return df, df_factors

df, df_factors = load_data()

# ---------------------------
# STREAMLIT APP
# ---------------------------
st.title("Predictive Maintenance Dashboard")

# ---- KPIs ----
total_machines = df['machine_udi'].nunique()
total_failures = df['failure_event'].sum()
failure_rate = total_failures / len(df) * 100

col1, col2, col3 = st.columns(3)
col1.metric("Total Machines", total_machines)
col2.metric("Failures (Last 100 Days)", total_failures)
col3.metric("Failure Rate (%)", f"{failure_rate:.2f}%")

st.markdown("_Failure rate calculated over the last 100 days of sensor data_")

# ---------------------------
# TOP 10 HIGH RISK MACHINES (Selectable)
# ---------------------------
top_risk_df = (
    df.groupby("machine_udi", as_index=False)
    .agg(avg_risk=("predicted_risk_score", "mean"),
         failures=("failure_event", "sum"))
    .sort_values(by="avg_risk", ascending=False)
    .head(10)
)

st.subheader("Select Machine from Top 10 High-Risk Machines")

top_10_machine_list = top_risk_df['machine_udi'].tolist()

# ---------------------------
# SESSION STATE INIT
# ---------------------------
if 'last_selector' not in st.session_state:
    st.session_state['last_selector'] = 'top10'

def on_top10_change():
    st.session_state['last_selector'] = 'top10'

def on_full_change():
    st.session_state['last_selector'] = 'full'

selected_from_top10 = st.selectbox(
    "Select Machine UDI (Top 10 High-Risk Machines):",
    options=top_10_machine_list,
    index=0,
    key='top10_selector',
    on_change=on_top10_change
)

full_machine_list = list(range(1, 101))

selected_from_full = st.selectbox(
    "Or select Machine UDI (Full List):",
    options=full_machine_list,
    index=0,
    key='full_selector',
    on_change=on_full_change
)

if st.session_state['last_selector'] == 'top10':
    selected_machine = st.session_state['top10_selector']
else:
    selected_machine = st.session_state['full_selector']

# ---------------------------
# FILTER DATA FOR SELECTED MACHINE
# ---------------------------
machine_data = df[df['machine_udi'] == selected_machine].sort_values('timestamp')
machine_factors = df_factors[df_factors['machine_udi'] == selected_machine].iloc[0]

# ---------------------------
# SENSOR HISTORY
# ---------------------------
st.subheader(f"Sensor History for Machine UDI {selected_machine}")

fig = go.Figure()
fig.add_trace(go.Scatter(x=machine_data['timestamp'], y=machine_data['tool_wear'], mode='lines', name='Tool wear (units)', line=dict(width=2)))
fig.add_trace(go.Scatter(x=machine_data['timestamp'], y=machine_data['process_temperature'], mode='lines', name='Process temperature (°C)', line=dict(width=2)))
fig.add_trace(go.Scatter(x=machine_data['timestamp'], y=machine_data['rotational_speed'], mode='lines', name='Rotational speed (RPM)', line=dict(width=2)))
fig.add_trace(go.Scatter(x=machine_data['timestamp'], y=machine_data['torque'], mode='lines', name='Torque (Nm)', line=dict(width=2)))

fail_events = machine_data[machine_data['failure_event'] == 1]
if not fail_events.empty:
    for failure_time in fail_events['timestamp']:
        fig.add_vline(x=failure_time, line=dict(color='red', dash='dash'), opacity=0.7)
else:
    st.info("No failure events recorded for this machine.")

fig.update_layout(xaxis_title="Timestamp", yaxis_title="Sensor Reading", hovermode="x unified", legend=dict(orientation='h'))
st.plotly_chart(fig, use_container_width=True)

# ---------------------------
# RISK AND STATUS
# ---------------------------
latest = machine_data.iloc[-1]
risk_score = latest['predicted_risk_score']

if risk_score < 0.3:
    risk_label = "Low Risk"
    risk_color = "green"
elif risk_score < 0.7:
    risk_label = "Medium Risk"
    risk_color = "orange"
else:
    risk_label = "High Risk"
    risk_color = "red"

failure_status = "No Failure" if latest['failure_event'] == 0 else "Failure"

col1, col2 = st.columns(2)
col1.markdown(f"### Failure Status: **{failure_status}**")
col2.markdown(
    f"### Risk Score: <span style='color:{risk_color}'><b>{risk_score:.2f} ({risk_label})</b></span>",
    unsafe_allow_html=True
)

# ---------------------------
# GAUGE CHART
# ---------------------------
st.subheader("Machine Risk Gauge")

gauge_fig = go.Figure(go.Indicator(
    mode="gauge+number",
    value=risk_score,
    gauge={
        "axis": {"range": [0, 1]},
        "steps": [
            {"range": [0, 0.3], "color": "lightgreen"},
            {"range": [0.3, 0.7], "color": "yellow"},
            {"range": [0.7, 1], "color": "red"}
        ],
        "threshold": {"line": {"color": "black", "width": 4}, "value": risk_score}
    },
    title={"text": "Machine Health Level"}
))

st.plotly_chart(gauge_fig, use_container_width=True)

# ---------------------------
# FACTOR IMPORTANCE - Bars + Metrics
# ---------------------------
st.subheader("Important Factors Influencing Risk")

factor_names = ['Tool wear', 'Process temperature', 'Rotational speed', 'Torque']
factor_values = [machine_factors[name] for name in factor_names]

fig_factors = go.Figure(go.Bar(
    x=factor_values,
    y=factor_names,
    orientation='h',
    marker=dict(color='steelblue')
))
st.plotly_chart(fig_factors, use_container_width=True)

cols = st.columns(len(factor_names))
for i, name in enumerate(factor_names):
    cols[i].metric(label=name, value=f"{machine_factors[name]:.2f}")

# ---------------------------
# MONTHLY FAILURE TREND (Selected Machine)
# ---------------------------
st.subheader(f"Monthly Failure Trend (Machine UDI {selected_machine})")

machine_monthly = machine_data.copy()
machine_monthly['month'] = machine_monthly['timestamp'].dt.to_period('M').astype(str)
monthly_failures = machine_monthly.groupby("month")['failure_event'].sum()

trend_fig = go.Figure()
trend_fig.add_trace(go.Scatter(
    x=monthly_failures.index,
    y=monthly_failures.values,
    mode='lines+markers',
    line=dict(width=3),
    marker=dict(size=8)
))
trend_fig.update_layout(xaxis_title="Month", yaxis_title="Failures")

st.plotly_chart(trend_fig, use_container_width=True)

# ---------------------------
# SHAP EXPLANATION
# ---------------------------
st.subheader(f"SHAP Explanation for Machine UDI {selected_machine}")

features = ['tool_wear', 'process_temperature', 'rotational_speed', 'torque']

X_latest = machine_data[features].iloc[-1:].copy()  # DataFrame with single row
X_train = df[features]
y_train = df['failure_event']

model = train_model(X_train, y_train)

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_latest)

expected_value = explainer.expected_value
if isinstance(expected_value, (list, np.ndarray)) and len(expected_value) > 1:
    expected_value = expected_value[0]

# SHAP values for the single sample are 1D array, extract them properly
shap_value_single = shap_values[0] if isinstance(shap_values, list) else shap_values
if shap_value_single.ndim > 1:
    shap_value_single = shap_value_single[0]

fig_shap, ax = plt.subplots(figsize=(8, 4))
shap.plots._waterfall.waterfall_legacy(
    expected_value,
    shap_value_single,
    feature_names=features,
    max_display=10,
    show=False
)
st.pyplot(fig_shap)

# ---------------------------
# DOWNLOAD BUTTON - Machine Data CSV
# ---------------------------
csv = machine_data.to_csv(index=False)
st.download_button(
    label="Download Selected Machine Data as CSV",
    data=csv,
    file_name=f'machine_{selected_machine}_data.csv',
    mime='text/csv'
)
