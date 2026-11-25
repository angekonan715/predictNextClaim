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


# The form no longer asks for these. An older saved model still expects them,
# and the usual value is what almost every policy had.
FILL_IF_MISSING = {
    "seats": 5,
    "municipality_type": "I",
}


def fitted_columns(model) -> list[str] | None:
    """Columns the saved model was trained on, when the file records them."""
    steps = getattr(model, "named_steps", None)
    if not steps:
        return None
    names = getattr(steps.get("preprocess"), "feature_names_in_", None)
    if names is None:
        return None
    return [str(name) for name in names]


def predict_frame(model_type: str, model, data: pd.DataFrame) -> np.ndarray:
    """Probability of at least one claim for each row already in `data`.

    The GLM adds log(exposure) as an offset. Every other model reads exposure
    as a column. Columns come from the saved model, not from the current
    settings, so a settings change cannot ask for a field the row does not have.
    """
    if model_type == "glm":
        offset = np.log(data[config.EXPOSURE_COL])
        return np.asarray(model.predict(data, offset=offset))

    columns = fitted_columns(model) or (config.FEATURES + [config.EXPOSURE_COL])
    return np.asarray(model.predict_proba(data.loc[:, columns])[:, 1])


def predict_claim_probability(bundle: dict, inputs: pd.DataFrame, exposure: float = 1.0) -> np.ndarray:
    """Probability of at least one claim over `exposure` years (default: a full year)."""
    frame = inputs.copy()
    model = bundle["model"]
    columns = fitted_columns(model)
    if columns is None:
        exposure_name = bundle.get("exposure_col", config.EXPOSURE_COL)
        columns = list(bundle.get("numeric_features", config.NUMERIC_FEATURES))
        columns += list(bundle.get("categorical_features", config.CATEGORICAL_FEATURES))
        if exposure_name not in columns:
            columns.append(exposure_name)
    exposure_name = bundle.get("exposure_col", config.EXPOSURE_COL)
    if exposure_name not in columns:
        exposure_name = next((name for name in columns if "exposure" in name), columns[-1])
    frame[exposure_name] = exposure
    for name in columns:
        if name not in frame.columns and name in FILL_IF_MISSING:
            frame[name] = FILL_IF_MISSING[name]
    return predict_frame(bundle["model_type"], model, frame)
