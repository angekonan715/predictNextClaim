"""Streamlit app: enter driver and vehicle details, see estimated claim probability.

Run from the project root:
    streamlit run app/app.py
"""
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

# Make `src` importable when Streamlit runs this file from app/
sys.path.append(str(Path(__file__).resolve().parents[1]))

from src import config  # noqa: E402
from src.evaluate import assign_risk_tier  # noqa: E402
from src.predict import load_model, predict_claim_probability  # noqa: E402

st.set_page_config(page_title="Claim Risk Estimator", page_icon="🚗", layout="centered")


@st.cache_resource
def get_model(model_mtime: float):
    # model_mtime is part of the cache key, so a replaced model file is loaded
    # on the next visit instead of the copy kept in memory.
    return load_model()


st.title("Car insurance claim risk")
st.write(
    "This estimates the chance that a driver files at least one claim in the next year. "
    "It was trained on real policies from a Spanish insurer, 2022–2023, and checked on 2024."
)
st.caption("Educational project. This is not an insurance quote.")

levels_file = config.PROCESSED_DIR / "feature_levels.json"
if not config.MODEL_FILE.exists() or not levels_file.exists():
    st.warning("Train the model first: `python -m src.data_prep` then `python -m src.train`.")
    st.stop()

bundle = get_model(config.MODEL_FILE.stat().st_mtime)
cutoffs = bundle.get("tier_cutoffs", config.RISK_TIER_CUTOFFS)

st.subheader("The model")
st.write(bundle.get("note", "The saved model estimates the probability of at least one claim."))
st.write(
    "The form uses facts an insurer would know at the start of the year: the driver, the car, "
    "the coverage, and where the car is driven. A shorter policy has less time to produce a claim, "
    "so the estimate below is for a full next year."
)

comparison_file = config.FIGURES_DIR / "model_comparison.csv"
chart_file = config.FIGURES_DIR / "model_comparison.png"
if comparison_file.exists():
    st.subheader("How the six models compared")
    st.write(
        "The models learned from 2022 and 2023, then took a test on 2024. "
        "AUC is a sorting score. Higher means the model puts drivers who really filed a claim above drivers who did not. "
        "The calibration gap checks the percentages: smaller means a predicted 20% chance really happens about 20% of the time."
    )
    st.dataframe(pd.read_csv(comparison_file, index_col=0), width="stretch")
    if chart_file.exists():
        st.image(str(chart_file), width="stretch")
    st.caption(
        "The orange bar is the model used below. The axis is zoomed in so small gaps are easier to see."
    )

FRIENDLY = {
    "bonus_score": {"G": "Good record", "N": "Neutral record", "B": "Poor record"},
    "policy_type": {
        "TP": "Third party",
        "TPG": "Third party plus glass",
        "CC": "Combined cover",
        "COMP_E": "Comprehensive with excess",
        "COMP_N": "Comprehensive, no excess",
    },
    "business_type": {"NB": "New policy", "P": "Existing policy"},
    "payment_frequency": {"A": "Once a year", "S": "Twice a year", "Q": "Every three months"},
    "fuel_type": {"D": "Diesel", "G": "Gasoline", "Unknown": "Unknown"},
    "municipality_type": {"I": "Inland", "C": "Coast", "IS": "Island"},
    "circulation_area": {"U": "Town", "R": "Country"},
}

levels = json.loads(levels_file.read_text(encoding="utf-8"))


def pick(column: str, title: str, help_text: str | None = None) -> str:
    """Show a plain-language choice and return the code stored in the data."""
    options = levels.get(column, [])
    labels = FRIENDLY.get(column, {})
    shown = [labels.get(code, code) for code in options]
    choice = st.selectbox(title, shown, help=help_text)
    return options[shown.index(choice)]


FACT_NAMES = {
    "total_exposure": "Time the policy was in force",
    "policy_type": "Coverage",
    "vehicle_value": "Vehicle value",
    "vehicle_age": "Vehicle age",
    "power_to_weight_ratio": "Weight per horsepower",
    "driver_age": "Driver age",
    "payment_frequency": "Payment schedule",
    "age_driving_licence": "Years with a licence",
    "vehicle_brand": "Brand",
    "fuel_type": "Fuel",
    "bonus_score": "Past claims",
    "circulation_area": "Town or country",
    "business_type": "New or existing policy",
    "municipality_type": "Home area",
    "seats": "Seats",
}


def feature_shares(bundle: dict) -> pd.DataFrame:
    """Share of LightGBM's splits that use each original fact.

    One-hot columns such as coverage=comprehensive are added back together
    so the chart shows Coverage, not one bar per coverage code.
    """
    fitted = bundle["model"].named_steps["model"]
    names = bundle["model"].named_steps["preprocess"].get_feature_names_out()
    gain = pd.Series(fitted.booster_.feature_importance(importance_type="gain"), index=names)
    columns = sorted(config.FEATURES + [config.EXPOSURE_COL], key=len, reverse=True)

    def original(name: str) -> str:
        rest = name.split("__", 1)[1]
        if name.startswith("num__"):
            return rest
        for column in columns:
            if rest == column or rest.startswith(column + "_"):
                return column
        return rest

    share = gain.groupby(gain.index.map(original)).sum()
    share = (share / share.sum()).sort_values(ascending=False)
    return pd.DataFrame({
        "Fact": [FACT_NAMES.get(name, name) for name in share.index],
        "Share": share.to_numpy(),
    })


with st.form("driver_info"):
    driver_col, car_col = st.columns(2)
    with driver_col:
        driver_age = st.slider("Driver age", 18, 90, 47)
        years_licensed = st.slider("Years with a licence", 0, 60, 20)
        bonus_score = pick(
            "bonus_score",
            "Past claims",
            "The insurer's record from before this year. Good means few past claims. "
            "Neutral is ordinary. Poor means a history of claims.",
        )
        business_type = pick("business_type", "Policy is")
        payment_frequency = pick(
            "payment_frequency",
            "Payment",
            "How often the premium is paid. In this data, paying more often goes with a higher claim rate. "
            "That is a clue about the customer, not proof that the payment plan causes a claim.",
        )
    with car_col:
        vehicle_age = st.slider("Vehicle age (years)", 0, 40, 24)
        value_cap = int(levels.get("vehicle_value_cap", 150000))
        vehicle_value = st.number_input("Vehicle value", 1000, max(value_cap, 1000), min(25000, value_cap), step=500)
        fuel_type = pick("fuel_type", "Fuel")
        vehicle_brand = pick("vehicle_brand", "Brand")
        seats = st.slider("Seats", 2, 9, 5)
    policy_type = pick("policy_type", "Coverage")
    place_col, area_col, power_col = st.columns(3)
    with place_col:
        municipality_type = pick("municipality_type", "Home area")
    with area_col:
        circulation_area = pick("circulation_area", "Usual driving")
    with power_col:
        power_to_weight_ratio = st.number_input("Weight per horsepower", 4.0, 40.0, 12.0, step=0.1)
    submitted = st.form_submit_button("Estimate my risk", width="stretch")

if submitted:
    user = pd.DataFrame([{
        "driver_age": driver_age,
        "vehicle_age": vehicle_age,
        "age_driving_licence": years_licensed,
        "vehicle_value": vehicle_value,
        "seats": seats,
        "power_to_weight_ratio": power_to_weight_ratio,
        "bonus_score": bonus_score,
        "policy_type": policy_type,
        "business_type": business_type,
        "payment_frequency": payment_frequency,
        "fuel_type": fuel_type,
        "vehicle_brand": vehicle_brand,
        "municipality_type": municipality_type,
        "circulation_area": circulation_area,
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

st.subheader("What the model uses most")
st.write(
    "Each bar is the share of the model's decisions that use that fact. "
    "Time on cover is the largest because a policy that was only active for part of the year "
    "has less time to claim. The estimate above is for a full year, so the other bars are what "
    "change from one driver to the next."
)
shares = feature_shares(bundle)
fig, ax = plt.subplots(figsize=(8, 5.2))
plot = shares.sort_values("Share", ascending=True)
ax.barh(plot["Fact"], plot["Share"], color="#4C78A8")
ax.set_xlabel("Share of the model's splits")
ax.set_xlim(0, float(plot["Share"].max()) * 1.18)
for i, value in enumerate(plot["Share"]):
    ax.text(value + 0.008, i, f"{value:.0%}", va="center", fontsize=9)
fig.tight_layout()
st.pyplot(fig)
plt.close(fig)
