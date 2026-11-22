"""Evaluation helpers used in notebooks/03_evaluation.ipynb."""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def gini(y_true, y_pred) -> float:
    """Normalized Gini coefficient = 2 * AUC - 1 (standard ranking metric in insurance)."""
    return 2 * roc_auc_score(y_true, y_pred) - 1


def lift_table(y_true, y_pred, exposure=None, n_bins: int = 10) -> pd.DataFrame:
    """Group policies into deciles of predicted risk and compare predicted vs. actual.

    Decile 1 = lowest predicted risk, decile n_bins = highest.
    """
    df = pd.DataFrame({"actual": np.asarray(y_true), "pred": np.asarray(y_pred)})
    df["exposure"] = 1.0 if exposure is None else np.asarray(exposure)
    df["decile"] = pd.qcut(df["pred"].rank(method="first"), n_bins, labels=range(1, n_bins + 1))
    out = df.groupby("decile", observed=True).agg(
        policies=("actual", "size"),
        exposure=("exposure", "sum"),
        predicted_rate=("pred", "mean"),
        actual_rate=("actual", "mean"),
    )
    out["lift"] = out["actual_rate"] / df["actual"].mean()
    return out.reset_index()


def calibration_table(y_true, y_pred, n_bins: int = 10) -> pd.DataFrame:
    """Mean predicted probability vs. observed claim rate within bins of predictions.

    A well-calibrated model has predicted_rate ≈ actual_rate in every bin.
    """
    df = pd.DataFrame({"actual": np.asarray(y_true), "pred": np.asarray(y_pred)})
    df["bin"] = pd.qcut(df["pred"].rank(method="first"), n_bins, labels=range(1, n_bins + 1))
    return (
        df.groupby("bin", observed=True)
        .agg(n=("actual", "size"), predicted_rate=("pred", "mean"), actual_rate=("actual", "mean"))
        .reset_index()
    )


def assign_risk_tier(prob, cutoffs: dict) -> np.ndarray:
    """Map probabilities to Low / Medium / High using cutoffs from config."""
    prob = np.asarray(prob)
    return np.where(prob < cutoffs["Low"], "Low",
                    np.where(prob < cutoffs["Medium"], "Medium", "High"))
