from __future__ import annotations

import json

import pandas as pd
import plotly.express as px
import streamlit as st

from utils import (
    FEATURE_COLUMNS,
    METRICS_PATH,
    MODEL_PATH,
    TARGET_COLUMN,
    build_feature_frame,
    load_model as load_saved_model,
    prepare_training_data,
    relief_plan,
    sieve_risk_category,
)


st.set_page_config(page_title="Silent Risk", layout="wide")


@st.cache_data
def load_dataset() -> pd.DataFrame:
    df = pd.read_csv("processed_burnout_data.csv")
    return prepare_training_data(df)


@st.cache_resource
def load_model():
    return load_saved_model(MODEL_PATH)


@st.cache_data
def load_metrics() -> dict:
    with open(METRICS_PATH, "r", encoding="utf-8") as handle:
        return json.load(handle)


def predict_score(model, feature_frame: pd.DataFrame) -> float:
    # Try with DataFrame first, fallback to list of dicts for FallbackBurnoutRegressor
    try:
        prediction = model.predict(feature_frame)[0]
    except TypeError:
        prediction = model.predict(feature_frame.to_dict(orient="records"))[0]
    return max(0.0, min(1.0, float(prediction)))


df = load_dataset()
model = load_model()
metrics = load_metrics()

# Title and introduction
st.title("🏗️ Silent Risk")
st.markdown("**An AI-Driven Predictive System for Corporate Burnout & Mental Health Intervention.**")
st.markdown("---")

st.sidebar.header("Burnout Predictor Simulator")
st.sidebar.markdown("Adjust the values below to simulate a real-time burnout prediction.")

sample_row = df.iloc[0]

# Sidebar inputs
designation = st.sidebar.slider("Designation (Level)", 0.0, 5.0, float(sample_row["Designation"]), 1.0)
resource_allocation = st.sidebar.slider(
    "Resource Allocation (Workload)", 1.0, 10.0, float(sample_row["Resource Allocation"]), 1.0
)
mental_fatigue = st.sidebar.slider(
    "Mental Fatigue Score", 0.0, 10.0, float(sample_row["Mental Fatigue Score"]), 0.1
)
tenure_days = st.sidebar.slider("Tenure Days", 0.0, 365.0, float(sample_row["Tenure_Days"]), 1.0)
workload_intensity = st.sidebar.slider(
    "Workload Intensity", 0.0, 4.0, float(sample_row["Workload_Intensity"]), 0.05
)

feature_frame = build_feature_frame(
    designation,
    resource_allocation,
    mental_fatigue,
    tenure_days,
    workload_intensity,
)

predicted_score = predict_score(model, feature_frame)
risk = sieve_risk_category(predicted_score)

# Dynamic Background Coloring CSS
bg_color = "#ffffff" # Default
text_color = "#000000"
if risk == "High Risk":
    bg_color = "#4D0000" # Deep red
    text_color = "#ffffff"
    accent = "🔴 Critical"
elif risk == "Medium Risk":
    bg_color = "#332A00" # Deep yellow/gold
    text_color = "#ffffff"
    accent = "🟡 Warning"
else:
    bg_color = "#003310" # Deep green
    text_color = "#ffffff"
    accent = "🟢 Healthy"

st.markdown(
    f"""
    <style>
    .stApp {{
        background-color: {bg_color};
        color: {text_color};
        transition: background-color 0.5s ease;
    }}
    h1, h2, h3, p, div {{
        color: {text_color} !important;
    }}
    .prediction-box {{
        padding: 2rem;
        border-radius: 10px;
        background: rgba(255, 255, 255, 0.1);
        margin-bottom: 2rem;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

st.header(f"Real-Time Prediction: {accent}")

col1, col2 = st.columns([1, 2])
with col1:
    st.markdown(f"### Burnout Score: {predicted_score:.2f} / 1.00")
    if risk == "High Risk":
        st.error(f"Status: {risk}")
    elif risk == "Medium Risk":
        st.warning(f"Status: {risk}")
    else:
        st.success(f"Status: {risk}")

with col2:
    st.markdown("### 🛠️ Actionable Solution Plan")
    st.markdown(f"**Recommendation:** {relief_plan(predicted_score, feature_frame.iloc[0].to_dict())}")


st.markdown("---")
st.subheader("Model Insights & Dataset")

col_a, col_b, col_c = st.columns(3)
col_a.metric("Employees Analyzed", len(df))
col_b.metric("Model RMSE", round(metrics["rmse"], 4))
col_c.metric("Model R²", round(metrics["r2"], 4))

left, right = st.columns(2)
with left:
    st.markdown("#### Feature Importance")
    st.markdown("This chart explains *why* the model predicts a certain way, building Data-Driven Empathy.")
    importance_df = (
        pd.DataFrame(
            {
                "Feature": list(metrics["feature_importance"].keys()),
                "Importance": list(metrics["feature_importance"].values()),
            }
        )
        .sort_values("Importance", ascending=True)
    )
    importance_fig = px.bar(
        importance_df,
        x="Importance",
        y="Feature",
        orientation="h",
        color="Importance",
        color_continuous_scale="Tealgrn",
    )
    importance_fig.update_layout(
        coloraxis_showscale=False, 
        height=320,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font_color=text_color
    )
    st.plotly_chart(importance_fig, use_container_width=True)

with right:
    st.markdown("#### Manager Risk Overview")
    preview_columns = ["Employee ID", "Burn Rate", "Risk_Zone"] + FEATURE_COLUMNS
    st.dataframe(df[preview_columns].head(10), use_container_width=True)
