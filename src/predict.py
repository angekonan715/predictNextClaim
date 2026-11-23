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


def predict_frame(model_type: str, model, data: pd.DataFrame) -> np.ndarray:
    """Probability of at least one claim for each row already in `data`.

    The GLM adds log(exposure) as an offset. Every other model reads exposure
    as a column. `data` must already contain the predictors and exposure.
    """
    if model_type == "glm":
        offset = np.log(data[config.EXPOSURE_COL])
        return np.asarray(model.predict(data, offset=offset))

    columns = config.FEATURES + [config.EXPOSURE_COL]
    return np.asarray(model.predict_proba(data[columns])[:, 1])


def predict_claim_probability(bundle: dict, inputs: pd.DataFrame, exposure: float = 1.0) -> np.ndarray:
    """Probability of at least one claim over `exposure` years (default: a full year)."""
    frame = inputs.copy()
    frame[bundle["exposure_col"]] = exposure
    return predict_frame(bundle["model_type"], bundle["model"], frame)
