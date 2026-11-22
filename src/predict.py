"""Load the saved model and predict claim probability for new inputs.

Used by the Streamlit app. Works for either the GLM or the LightGBM bundle
saved by src/train.py.
"""
import joblib
import numpy as np
import pandas as pd

from src import config


def load_model(path=config.MODEL_FILE) -> dict:
    return joblib.load(path)


def predict_claim_probability(bundle: dict, inputs: pd.DataFrame, exposure: float = 1.0) -> np.ndarray:
    """Probability of at least one claim over `exposure` years (default: a full year)."""
    X = inputs.copy()
    X[bundle["exposure_col"]] = exposure

    if bundle["model_type"] == "glm":
        offset = np.log(X[bundle["exposure_col"]])
        return np.asarray(bundle["model"].predict(X, offset=offset))

    cols = bundle["numeric_features"] + bundle["categorical_features"] + [bundle["exposure_col"]]
    return bundle["model"].predict_proba(X[cols])[:, 1]
