"""Project settings: paths, column names and modeling choices.

Update the column names below once you have read the data dictionary.
Every other script imports from here, so you only change things in one place.
"""
from pathlib import Path

# --- Paths ---------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
FIGURES_DIR = ROOT / "reports" / "figures"

RAW_DATA_FILE = "Dataset of motor insurance portfolio.csv"
TRAIN_FILE = PROCESSED_DIR / "train.csv"
TEST_FILE = PROCESSED_DIR / "test.csv"
MODEL_FILE = MODELS_DIR / "claim_model.joblib"

# --- Key columns (rename to match the real dataset) ----------------------
POLICY_ID_COL = "insured_id"
YEAR_COL = "year"
EXPOSURE_COL = "total_exposure"  # fraction of the year insured (0–1)
CLAIM_COUNT_COL = "total_claims"  # all coverages; target is 1 if this is > 0
TARGET = "claim"                 # created in data_prep.make_target()

# --- Predictors ----------------------------------------------------------
# TODO: choose 10–15 predictors after EDA. Use names AFTER cleaning/feature creation.
# Names must be columns in the training file. This dataset has no
# years_licensed or vehicle_type column (licence tenure is age_driving_licence).
NUMERIC_FEATURES = [
    "driver_age",
    "vehicle_age",
    "vehicle_value",
]
CATEGORICAL_FEATURES = []
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# --- Validation design ---------------------------------------------------
TRAIN_YEARS = [2022, 2023]
TEST_YEARS = [2024]

RANDOM_STATE = 42

# --- App -----------------------------------------------------------------
# Full-year LightGBM probabilities on the 2024 test, split into thirds.
# Low is the lower third, Medium the middle third, High the upper third.
RISK_TIER_CUTOFFS = {"Low": 0.166, "Medium": 0.196}  # above Medium cutoff = High
