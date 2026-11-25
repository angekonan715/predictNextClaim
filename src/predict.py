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


def model_columns(model, bundle: dict | None = None) -> list[str]:
    """Columns the saved model reads. Never the live settings list.

    The live settings can still name a field the form no longer sends. The
    lists stored on the fitted preprocessor are the ones that were used to train.
    """
    found: list[str] = []
    steps = getattr(model, "named_steps", None)
    preprocess = steps.get("preprocess") if steps else None
    specs = []
    if preprocess is not None:
        specs = list(getattr(preprocess, "transformers_", None) or getattr(preprocess, "transformers", []))
    for _name, _transformer, cols in specs:
        if isinstance(cols, (list, tuple)):
            found.extend(str(col) for col in cols)
    if not found and bundle is not None:
        found.extend(str(col) for col in bundle.get("numeric_features", config.NUMERIC_FEATURES))
        found.extend(str(col) for col in bundle.get("categorical_features", config.CATEGORICAL_FEATURES))
        found.append(str(bundle.get("exposure_col", config.EXPOSURE_COL)))
    ordered: list[str] = []
    for name in found:
        if name not in ordered:
            ordered.append(name)
    return ordered


def _category_defaults(model) -> dict[str, object]:
    """First trained category for each categorical column, used only as a fill."""
    steps = getattr(model, "named_steps", None)
    preprocess = steps.get("preprocess") if steps else None
    if preprocess is None:
        return {}
    specs = getattr(preprocess, "transformers_", None) or []
    defaults = {}
    for _name, transformer, cols in specs:
        categories = getattr(transformer, "categories_", None)
        if categories is None or not isinstance(cols, (list, tuple)):
            continue
        for column, levels in zip(cols, categories):
            if len(levels):
                defaults[str(column)] = levels[0]
    return defaults


def _prepare(model, data: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Return a frame with exactly the model's columns, filled where the form has no value."""
    frame = data.copy()
    categories = _category_defaults(model)
    for name in columns:
        if name in frame.columns:
            continue
        if name in FILL_IF_MISSING:
            frame[name] = FILL_IF_MISSING[name]
        elif name in categories:
            frame[name] = categories[name]
        else:
            frame[name] = 0
    return frame.loc[:, columns]


def predict_frame(model_type: str, model, data: pd.DataFrame) -> np.ndarray:
    """Probability of at least one claim for each row already in `data`.

    The GLM adds log(exposure) as an offset. Every other model reads exposure
    as a column.
    """
    if model_type == "glm":
        offset = np.log(data[config.EXPOSURE_COL].astype(float))
        return np.asarray(model.predict(data, offset=offset))

    steps = getattr(model, "named_steps", None)
    preprocess = steps.get("preprocess") if steps else None
    estimator = steps.get("model") if steps else None
    names: list[str] = []
    fitted = getattr(preprocess, "feature_names_in_", None) if preprocess is not None else None
    if fitted is not None:
        names.extend(str(name) for name in fitted)
    for name in model_columns(model):
        if name not in names:
            names.append(name)
    prepared = _prepare(model, data, names) if names else data
    if preprocess is not None and estimator is not None:
        # The hosted app's LightGBM build looks column names up on a table.
        # Hand it a plain numeric array, which is what this model was trained on.
        transformed = preprocess.transform(prepared)
        if hasattr(transformed, "toarray"):
            transformed = transformed.toarray()
        transformed = np.asarray(transformed, dtype=float)
        return np.asarray(estimator.predict_proba(transformed)[:, 1])
    return np.asarray(model.predict_proba(prepared)[:, 1])


def predict_claim_probability(bundle: dict, inputs: pd.DataFrame, exposure: float = 1.0) -> np.ndarray:
    """Probability of at least one claim over `exposure` years (default: a full year)."""
    frame = inputs.copy()
    model = bundle["model"]
    columns = model_columns(model, bundle)
    exposure_name = str(bundle.get("exposure_col", config.EXPOSURE_COL))
    if exposure_name not in columns:
        exposure_name = next((name for name in columns if "exposure" in name), exposure_name)
    frame[exposure_name] = exposure
    if columns and exposure_name not in columns:
        columns = [*columns, exposure_name]
    return predict_frame(bundle["model_type"], model, _prepare(model, frame, columns) if columns else frame)
