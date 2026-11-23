"""Load, clean and split the motor insurance data.

Run from the project root:
    python -m src.data_prep
"""
import json

import pandas as pd

from src import config


def load_raw(file_name: str | None = None) -> pd.DataFrame:
    """Read the downloaded dataset (Excel or CSV)."""
    if file_name is None:
        file_name = config.RAW_DATA_FILE
    path = config.RAW_DIR / file_name
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Download the data (see data/README.md) "
            "and set RAW_DATA_FILE in src/config.py."
        )
    if path.suffix in (".xlsx", ".xls"):
        return pd.read_excel(path)
    # The Mendeley export separates fields with semicolons.
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        header = handle.readline()
    sep = ";" if header.count(";") > header.count(",") else ","
    return pd.read_csv(path, sep=sep)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows the model cannot use, and fill the one missing category.

    Exposure must be above zero because the GLM adds log(exposure). A missing
    age or vehicle value has no honest fill, and there are only a few hundred
    of those rows. Missing fuel is kept as its own group, "Unknown".
    """
    df = df.copy()
    before = len(df)

    df = df.loc[df[config.EXPOSURE_COL] > 0]
    required = [
        "driver_age",
        "vehicle_age",
        "age_driving_licence",
        "vehicle_value",
        "seats",
        "power_to_weight_ratio",
    ]
    df = df.dropna(subset=required)
    df = df.loc[
        (df["driver_age"] >= 16)
        & (df["vehicle_age"] >= 0)
        & (df["age_driving_licence"] >= 0)
        & (df["power_to_weight_ratio"] > 0)
    ]
    df["fuel_type"] = df["fuel_type"].fillna("Unknown")

    print(f"Clean: kept {len(df):,} of {before:,} rows")
    return df


def cap_vehicle_value(train: pd.DataFrame, test: pd.DataFrame):
    """Cap declared value at the training-year 99th percentile.

    The cap is learned on 2022–2023 only, then applied to 2024, so the test
    year does not help choose the cutoff. One very expensive car otherwise
    pulls the rate away from a typical car.
    """
    cap = float(train["vehicle_value"].quantile(0.99))
    train = train.copy()
    test = test.copy()
    train["vehicle_value"] = train["vehicle_value"].clip(upper=cap)
    test["vehicle_value"] = test["vehicle_value"].clip(upper=cap)
    print(f"Vehicle value capped at {cap:,.0f}")
    return train, test, cap


def group_rare_brands(train: pd.DataFrame, test: pd.DataFrame):
    """Fold uncommon brands into Other, using training-year counts only."""
    counts = train["vehicle_brand"].value_counts()
    keep = set(counts[counts >= config.MIN_BRAND_COUNT].index)
    train = train.copy()
    test = test.copy()
    train["vehicle_brand"] = train["vehicle_brand"].where(
        train["vehicle_brand"].isin(keep), "Other"
    )
    test["vehicle_brand"] = test["vehicle_brand"].where(
        test["vehicle_brand"].isin(keep), "Other"
    )
    print(f"Vehicle brands kept: {len(keep)} common, the rest grouped as Other")
    return train, test, keep


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


def save_feature_levels(train: pd.DataFrame, value_cap: float) -> None:
    """Save the choices the app can offer, in order of how common they are."""
    levels = {"vehicle_value_cap": value_cap}
    for column in config.CATEGORICAL_FEATURES:
        levels[column] = train[column].value_counts().index.astype(str).tolist()
    config.LEVELS_FILE.write_text(json.dumps(levels, indent=2), encoding="utf-8")


def main():
    df = load_raw()
    print(f"Raw data: {df.shape[0]:,} rows, {df.shape[1]} columns")

    df = clean(df)
    df = make_target(df)
    train, test = split_by_year(df)
    train, test, value_cap = cap_vehicle_value(train, test)
    train, test, _keep = group_rare_brands(train, test)
    train = keep_model_columns(train)
    test = keep_model_columns(test)

    print(f"Train: {len(train):,} rows | claim rate {train[config.TARGET].mean():.2%}")
    print(f"Test:  {len(test):,} rows | claim rate {test[config.TARGET].mean():.2%}")

    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    train.to_csv(config.TRAIN_FILE, index=False)
    test.to_csv(config.TEST_FILE, index=False)
    save_feature_levels(train, value_cap)
    print(f"Saved to {config.PROCESSED_DIR}")


if __name__ == "__main__":
    main()
