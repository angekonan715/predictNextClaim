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

RAW_DATA_FILE = "TODO_set_downloaded_file_name.xlsx"  # TODO: e.g. "motor_policies.xlsx"
TRAIN_FILE = PROCESSED_DIR / "train.csv"
TEST_FILE = PROCESSED_DIR / "test.csv"
MODEL_FILE = MODELS_DIR / "claim_model.joblib"

# --- Key columns (rename to match the real dataset) ----------------------
POLICY_ID_COL = "policy_id"      # TODO
YEAR_COL = "year"                # TODO: the policy year
EXPOSURE_COL = "exposure"        # TODO: fraction of the year insured (0–1)
CLAIM_COUNT_COL = "n_claims"     # TODO: number of claims (choose which coverage)
TARGET = "claim"                 # created in data_prep.make_target()

# --- Predictors ----------------------------------------------------------
# TODO: choose 10–15 predictors after EDA. Use names AFTER cleaning/feature creation.
NUMERIC_FEATURES = [
    "driver_age",
    "years_licensed",
    "vehicle_age",
    "vehicle_value",
]
CATEGORICAL_FEATURES = [
    "vehicle_type",
]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# --- Validation design ---------------------------------------------------
TRAIN_YEARS = [2022, 2023]
TEST_YEARS = [2024]

RANDOM_STATE = 42

# --- App -----------------------------------------------------------------
# TODO: set these from your Task 4 results (e.g. tier boundaries on the test set).
RISK_TIER_CUTOFFS = {"Low": 0.05, "Medium": 0.12}  # above Medium cutoff = High
