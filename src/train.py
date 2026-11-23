"""Fit the claim models and save the one chosen for the app.

Run from the project root (after src.data_prep):
    python -m src.train

Every model answers the same question: the probability of at least one claim.
They all use driver age, vehicle age, vehicle value, and exposure, and they
are all fit on 2022–2023 only. The 2024 file is used to compare them.
"""
import joblib
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from src import config
from src.evaluate import choose_model, comparison_table
from src.predict import predict_frame


# Plain-language descriptions. The app shows the one for the saved model.
MODEL_NOTES = {
    "glm": (
        "The GLM is a formula for the chance of at least one claim. "
        "Vehicle value and vehicle age shift that chance by a fixed percentage, "
        "and log(exposure) shortens the chance when the policy covered only part of the year."
    ),
    "logistic": (
        "Logistic regression draws one straight line for the log-odds of a claim. "
        "Each predictor adds or subtracts a fixed amount. Inputs are scaled so vehicle value, "
        "which is in the tens of thousands, does not drown ages, which are in the tens."
    ),
    "tree": (
        "A decision tree asks a few yes-or-no questions, such as whether vehicle value "
        "is above a cutoff, and groups policies that follow the same answers. "
        "The prediction is the claim rate of that group. The tree is limited to four questions deep."
    ),
    "forest": (
        "A random forest grows many decision trees, each on a random sample of policies, "
        "and averages their probabilities. One tree can latch onto a quirk. The average is steadier."
    ),
    "xgboost": (
        "XGBoost builds trees one after another. Each new tree corrects the claims "
        "the earlier trees got wrong. The final probability is the sum of those corrections."
    ),
    "lgbm": (
        "LightGBM estimates the chance of at least one claim by adding hundreds of small decision trees. "
        "Each new tree corrects the policies the earlier trees scored too high or too low. "
        "It uses the driver, the car, the coverage, where the car is driven, and how much of the year the policy was in force."
    ),
}

DISPLAY_NAMES = {
    "glm": "GLM",
    "logistic": "Logistic regression",
    "tree": "Decision tree",
    "forest": "Random forest",
    "xgboost": "XGBoost",
    "lgbm": "LightGBM",
}


def usable_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows that cannot be scored.

    A missing predictor has no place in the formula, and log(exposure) is
    undefined when exposure is zero.
    """
    needed = [config.TARGET, config.EXPOSURE_COL] + config.FEATURES
    data = df.dropna(subset=needed)
    return data.loc[data[config.EXPOSURE_COL] > 0].copy()


def glm_formula() -> str:
    """Build the GLM formula from the feature lists in config."""
    terms = list(config.NUMERIC_FEATURES) + [f"C({c})" for c in config.CATEGORICAL_FEATURES]
    return f"{config.TARGET} ~ " + " + ".join(terms)


def fit_glm(train: pd.DataFrame):
    """Binomial GLM with a cloglog link and a log(exposure) offset.

    If claims arrive at rate lambda, the chance of at least one claim over
    exposure t is 1 - exp(-lambda * t). The cloglog link turns that into
    log(lambda) + log(t), so log(exposure) is added rather than estimated.
    """
    data = usable_rows(train)
    model = smf.glm(
        formula=glm_formula(),
        data=data,
        family=sm.families.Binomial(link=sm.families.links.CLogLog()),
        offset=np.log(data[config.EXPOSURE_COL]),
    )
    return model.fit()


def _pipeline(model, scale_numeric: bool) -> Pipeline:
    """Turn raw columns into the matrix the classifier expects."""
    numeric = config.NUMERIC_FEATURES + [config.EXPOSURE_COL]
    number_step = StandardScaler() if scale_numeric else "passthrough"
    transformers = [("num", number_step, numeric)]
    if config.CATEGORICAL_FEATURES:
        transformers.append(
            ("cat", OneHotEncoder(handle_unknown="ignore"), config.CATEGORICAL_FEATURES)
        )
    return Pipeline([
        ("preprocess", ColumnTransformer(transformers)),
        ("model", model),
    ])


def fit_classifier(train: pd.DataFrame, model, scale_numeric: bool = False) -> Pipeline:
    """Fit a yes/no classifier. Exposure is a column, not an offset."""
    data = usable_rows(train)
    pipe = _pipeline(model, scale_numeric)
    pipe.fit(data[config.FEATURES + [config.EXPOSURE_COL]], data[config.TARGET])
    return pipe


def fit_logistic(train: pd.DataFrame) -> Pipeline:
    """Logistic regression. Inputs are scaled so the coefficients can be compared."""
    model = LogisticRegression(max_iter=500, random_state=config.RANDOM_STATE)
    return fit_classifier(train, model, scale_numeric=True)


def fit_tree(train: pd.DataFrame) -> Pipeline:
    """One decision tree, four questions deep, with at least 200 policies in each leaf."""
    model = DecisionTreeClassifier(
        max_depth=4,
        min_samples_leaf=200,
        random_state=config.RANDOM_STATE,
    )
    return fit_classifier(train, model)


def fit_forest(train: pd.DataFrame) -> Pipeline:
    """200 trees, each at most 8 questions deep. The probability is their average."""
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        min_samples_leaf=100,
        n_jobs=-1,
        random_state=config.RANDOM_STATE,
    )
    return fit_classifier(train, model)


def fit_xgboost(train: pd.DataFrame) -> Pipeline:
    """300 boosted trees. Each tree is shallow so one tree cannot dominate."""
    model = XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=4,
        min_child_weight=200,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=config.RANDOM_STATE,
        n_jobs=-1,
        verbosity=0,
    )
    return fit_classifier(train, model)


def fit_lgbm(train: pd.DataFrame) -> Pipeline:
    """300 boosted trees, LightGBM's version of the same idea as XGBoost."""
    model = LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=31,
        min_child_samples=200,
        random_state=config.RANDOM_STATE,
        verbose=-1,
    )
    return fit_classifier(train, model)


def fit_all(train: pd.DataFrame) -> dict:
    """Fit every model on the same training rows."""
    return {
        "glm": fit_glm(train),
        "logistic": fit_logistic(train),
        "tree": fit_tree(train),
        "forest": fit_forest(train),
        "xgboost": fit_xgboost(train),
        "lgbm": fit_lgbm(train),
    }


def score_models(models: dict, test: pd.DataFrame) -> pd.DataFrame:
    """AUC, Gini, Brier, and calibration on the rows passed in (the 2024 test)."""
    data = usable_rows(test)
    predictions = {
        name: predict_frame(name, model, data) for name, model in models.items()
    }
    return comparison_table(data[config.TARGET], predictions)


def save_model(model, model_type: str, tier_cutoffs: dict):
    """Save the model with what the app needs to use it."""
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    bundle = {
        "model_type": model_type,
        "model": model,
        "numeric_features": config.NUMERIC_FEATURES,
        "categorical_features": config.CATEGORICAL_FEATURES,
        "exposure_col": config.EXPOSURE_COL,
        "note": MODEL_NOTES[model_type],
        "tier_cutoffs": tier_cutoffs,
    }
    joblib.dump(bundle, config.MODEL_FILE)
    print(f"Saved {DISPLAY_NAMES[model_type]} to {config.MODEL_FILE}")


def _tier_cutoffs(model_type: str, model, test: pd.DataFrame) -> dict:
    """Low / Medium / High are the lower, middle, and upper third of next-year scores.

    The app asks about a full year, so the cutoffs use exposure 1, not the
    partial year stored on each test row.
    """
    full_year = usable_rows(test)
    full_year[config.EXPOSURE_COL] = 1.0
    probabilities = predict_frame(model_type, model, full_year)
    low, medium = np.quantile(probabilities, [1 / 3, 2 / 3])
    return {"Low": float(low), "Medium": float(medium)}


def save_comparison(shown: pd.DataFrame) -> None:
    """Write the 2024 comparison table and bar chart for the notebook and the app."""
    import matplotlib.pyplot as plt

    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    table = shown.round(3).sort_values("AUC", ascending=False)
    table.to_csv(config.FIGURES_DIR / "model_comparison.csv")

    plot_df = table.sort_values("AUC", ascending=True)
    colors = ["#F58518" if name == "LightGBM" else "#4C78A8" for name in plot_df.index]
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.barh(plot_df.index.astype(str), plot_df["AUC"], color=colors)
    span = float(plot_df["AUC"].max() - plot_df["AUC"].min())
    ax.set_xlim(float(plot_df["AUC"].min()) - max(0.02, span), float(plot_df["AUC"].max()) + 0.015)
    ax.set_xlabel("Sorting score on 2024 (higher is better)")
    ax.set_title("Which model is best at sorting who will file a claim")
    for i, value in enumerate(plot_df["AUC"]):
        ax.text(value + 0.001, i, f"{value:.3f}", va="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / "model_comparison.png", dpi=150)
    plt.close(fig)


def main():
    train = pd.read_csv(config.TRAIN_FILE)
    test = pd.read_csv(config.TEST_FILE)
    print(f"Training on {len(usable_rows(train)):,} rows from 2022–2023")

    models = fit_all(train)
    scores = score_models(models, test)
    shown = scores.rename(index=DISPLAY_NAMES)
    print(shown.round(3).to_string())
    save_comparison(shown)

    winner = choose_model(scores)
    print(f"\nChosen for the app: {DISPLAY_NAMES[winner]}")
    print(MODEL_NOTES[winner])

    cutoffs = _tier_cutoffs(winner, models[winner], test)
    config.RISK_TIER_CUTOFFS = cutoffs
    print(f"Risk tiers (full next year): Low below {cutoffs['Low']:.1%}, "
          f"Medium below {cutoffs['Medium']:.1%}, High above that.")
    save_model(models[winner], winner, cutoffs)


if __name__ == "__main__":
    main()
