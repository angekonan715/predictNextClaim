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


st.title("Car insurance claim risk")
st.write(
    "This estimates the chance that a driver files at least one claim in the next year. "
    "It was trained on real policies from a Spanish insurer, 2022–2023, and checked on 2024."
)
st.caption("Educational project. This is not an insurance quote.")

if not config.MODEL_FILE.exists():
    st.warning("No trained model found. Run `python -m src.train` first.")
    st.stop()

bundle = get_model()
cutoffs = bundle.get("tier_cutoffs", config.RISK_TIER_CUTOFFS)

st.subheader("The model")
st.write(bundle.get("note", "The saved model estimates the probability of at least one claim."))
st.write(
    "On the 2024 policies, how much of the year the policy was in force explains most of the "
    "ranking, because a shorter policy has less time to produce a claim. For a full next year "
    "the three inputs below are what move the number: vehicle value first, then vehicle age, then driver age."
)

with st.form("driver_info"):
    driver_age = st.slider("Driver age", 18, 90, 47)
    vehicle_age = st.slider("Vehicle age (years)", 0, 40, 24)
    vehicle_value = st.number_input("Vehicle value", 1000, 150000, 25000, step=500)
    submitted = st.form_submit_button("Estimate my risk", use_container_width=True)

if submitted:
    user = pd.DataFrame([{
        "driver_age": driver_age,
        "vehicle_age": vehicle_age,
        "vehicle_value": vehicle_value,
    }])

    prob = float(predict_claim_probability(bundle, user)[0])
    tier = assign_risk_tier([prob], cutoffs)[0]

    st.divider()
    m1, m2 = st.columns(2)
    m1.metric("Chance of a claim next year", f"{prob:.1%}")
    m2.metric("Compared with other policies", tier)

    st.write(
        f"Low is under {cutoffs['Low']:.1%}, Medium is under {cutoffs['Medium']:.1%}, "
        "and High is above that. Those lines are the lower, middle, and upper third of "
        "full-year predictions on the 2024 policies."
    )
