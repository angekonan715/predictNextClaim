"""Streamlit app: enter driver and vehicle details, see estimated claim probability.

Run from the project root:
    streamlit run app/app.py
"""
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# Make `src` importable when Streamlit runs this file from app/
sys.path.append(str(Path(__file__).resolve().parents[1]))

from src import config  # noqa: E402
from src.evaluate import assign_risk_tier  # noqa: E402
from src.predict import load_model, predict_claim_probability  # noqa: E402

st.set_page_config(page_title="Claim Risk Estimator", page_icon="🚗", layout="centered")


@st.cache_resource
def get_model():
    return load_model()


st.title("🚗 Car Insurance Claim Risk Estimator")
st.write(
    "Estimate the probability that a driver files at least one car insurance claim "
    "in the next year, based on a model trained on real policy data (Spain, 2022–2024)."
)
st.caption("Educational project. This is not an insurance quote.")

if not config.MODEL_FILE.exists():
    st.warning("No trained model found. Run `python -m src.train` first.")
    st.stop()

bundle = get_model()

# TODO: make these inputs match config.FEATURES exactly (names, units, categories).
# Only ask for things a normal person knows about themselves and their car.
with st.form("driver_info"):
    col1, col2 = st.columns(2)
    with col1:
        driver_age = st.slider("Driver age", 18, 90, 35)
        years_licensed = st.slider("Years with a driver's license", 0, 70, 10)
    with col2:
        vehicle_age = st.slider("Vehicle age (years)", 0, 30, 5)
        vehicle_value = st.number_input("Vehicle value (€)", 1000, 150000, 15000, step=500)
    vehicle_type = st.selectbox("Vehicle type", ["TODO: categories from data"])
    submitted = st.form_submit_button("Estimate my risk", use_container_width=True)

if submitted:
    user = pd.DataFrame([{
        "driver_age": driver_age,
        "years_licensed": years_licensed,
        "vehicle_age": vehicle_age,
        "vehicle_value": vehicle_value,
        "vehicle_type": vehicle_type,
    }])

    prob = float(predict_claim_probability(bundle, user)[0])
    tier = assign_risk_tier([prob], config.RISK_TIER_CUTOFFS)[0]

    st.divider()
    m1, m2 = st.columns(2)
    m1.metric("Probability of a claim next year", f"{prob:.1%}")
    m2.metric("Risk tier", tier)

    # TODO: add context, e.g. "The average driver in the data has a X% claim rate."
    # TODO (optional): show which inputs pushed the risk up or down (GLM relativities or SHAP).
