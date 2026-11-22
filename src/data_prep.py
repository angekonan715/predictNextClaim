"""Load, clean and split the motor insurance data.

Run from the project root:
    python -m src.data_prep

Each function has TODOs for decisions you need to make after reading the
data dictionary. Document every decision in notebooks/01_eda.ipynb.
"""
import numpy as np
import pandas as pd

from src import config


def load_raw(file_name: str = config.RAW_DATA_FILE) -> pd.DataFrame:
    """Read the downloaded dataset (Excel or CSV)."""
    path = config.RAW_DIR / file_name
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Download the data (see data/README.md) "
            "and set RAW_DATA_FILE in src/config.py."
        )
    if path.suffix in (".xlsx", ".xls"):
        return pd.read_excel(path)
    return pd.read_csv(path)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Fix types, rename columns, handle missing and impossible values."""
    df = df.copy()

    # TODO: rename Spanish/original columns to clear English names, e.g.
    # df = df.rename(columns={"Fecha_nacimiento": "birth_date", ...})

    # TODO: convert date columns
    # for col in ["birth_date", "license_date", "vehicle_registration_date"]:
    #     df[col] = pd.to_datetime(df[col], dayfirst=True, errors="coerce")

    # TODO: handle missing values (drop, impute, or add an "unknown" category).
    # Record how many rows each step removes.

    # TODO: remove impossible values (e.g. negative exposure, driver age < 18)

    return df


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create model features from the cleaned columns."""
    df = df.copy()

    # TODO: derive ages from dates relative to the start of each policy year, e.g.
    # year_start = pd.to_datetime(df[config.YEAR_COL].astype(str) + "-01-01")
    # df["driver_age"] = (year_start - df["birth_date"]).dt.days / 365.25
    # df["years_licensed"] = (year_start - df["license_date"]).dt.days / 365.25
    # df["vehicle_age"] = ...

    # TODO: consider capping extreme values (e.g. vehicle_value at the 99th percentile)
    # and grouping rare categories into "Other".

    return df


def make_target(df: pd.DataFrame) -> pd.DataFrame:
    """Binary target: 1 if the policy had at least one claim in the year."""
    df = df.copy()
    df[config.TARGET] = (df[config.CLAIM_COUNT_COL] > 0).astype(int)
    return df


def split_by_year(df: pd.DataFrame):
    """Out-of-time split: train on earlier years, test on the latest year."""
    train = df[df[config.YEAR_COL].isin(config.TRAIN_YEARS)].copy()
    test = df[df[config.YEAR_COL].isin(config.TEST_YEARS)].copy()
    return train, test


def keep_model_columns(df: pd.DataFrame) -> pd.DataFrame:
    cols = [config.POLICY_ID_COL, config.YEAR_COL, config.EXPOSURE_COL,
            config.TARGET] + config.FEATURES
    return df[[c for c in cols if c in df.columns]]


def main():
    df = load_raw()
    print(f"Raw data: {df.shape[0]:,} rows, {df.shape[1]} columns")

    df = clean(df)
    df = add_features(df)
    df = make_target(df)
    df = keep_model_columns(df)

    train, test = split_by_year(df)
    print(f"Train: {len(train):,} rows | claim rate {train[config.TARGET].mean():.2%}")
    print(f"Test:  {len(test):,} rows | claim rate {test[config.TARGET].mean():.2%}")

    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    train.to_csv(config.TRAIN_FILE, index=False)
    test.to_csv(config.TEST_FILE, index=False)
    print(f"Saved to {config.PROCESSED_DIR}")


if __name__ == "__main__":
    main()
