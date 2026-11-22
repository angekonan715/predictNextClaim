"""Fit the GLM and LightGBM models and save the chosen one for the app.

Run from the project root (after src.data_prep):
    python -m src.train
"""
import joblib
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from src import config


def glm_formula() -> str:
    """Build the GLM formula from the feature lists in config."""
    terms = list(config.NUMERIC_FEATURES) + [f"C({c})" for c in config.CATEGORICAL_FEATURES]
    # TODO: consider bands or splines for non-linear effects, e.g. bs(driver_age, df=4)
    return f"{config.TARGET} ~ " + " + ".join(terms)


def fit_glm(train: pd.DataFrame):
    """Binomial GLM with complementary log-log link and log(exposure) offset.

    With a cloglog link, P(claim) = 1 - exp(-exposure * exp(x'b)), which is the
    probability of at least one claim if claims follow a Poisson process. That is
    why it handles policies insured for only part of the year correctly.
    """
    model = smf.glm(
        formula=glm_formula(),
        data=train,
        family=sm.families.Binomial(link=sm.families.links.CLogLog()),
        offset=np.log(train[config.EXPOSURE_COL]),
    )
    return model.fit()


def fit_lgbm(train: pd.DataFrame) -> Pipeline:
    """LightGBM classifier with one-hot encoding for categorical features.

    Exposure is included as a feature so the model can learn that shorter
    policies have fewer claims. For app predictions, exposure is set to 1.0.
    """
    preprocess = ColumnTransformer(
        transformers=[
            ("num", "passthrough", config.NUMERIC_FEATURES + [config.EXPOSURE_COL]),
            ("cat", OneHotEncoder(handle_unknown="ignore"), config.CATEGORICAL_FEATURES),
        ]
    )
    # TODO: tune with cross-validation on the training years only
    clf = LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=31,
        min_child_samples=200,
        random_state=config.RANDOM_STATE,
        verbose=-1,
    )
    pipe = Pipeline([("preprocess", preprocess), ("model", clf)])
    X = train[config.FEATURES + [config.EXPOSURE_COL]]
    pipe.fit(X, train[config.TARGET])
    return pipe


def save_model(model, model_type: str):
    """Save the model with what the app needs to use it."""
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    bundle = {
        "model_type": model_type,          # "glm" or "lgbm"
        "model": model,
        "numeric_features": config.NUMERIC_FEATURES,
        "categorical_features": config.CATEGORICAL_FEATURES,
        "exposure_col": config.EXPOSURE_COL,
    }
    joblib.dump(bundle, config.MODEL_FILE)
    print(f"Saved {model_type} model to {config.MODEL_FILE}")


def main():
    train = pd.read_csv(config.TRAIN_FILE)
    print(f"Training on {len(train):,} rows")

    glm = fit_glm(train)
    print(glm.summary())

    lgbm = fit_lgbm(train)
    print("LightGBM fitted.")

    # TODO: after Task 4, choose the final model for the app and explain why.
    final_choice = "glm"
    save_model(glm if final_choice == "glm" else lgbm, final_choice)


if __name__ == "__main__":
    main()
