"""Evaluation helpers used in notebooks/03_evaluation.ipynb."""
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, roc_auc_score


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


def comparison_table(y_true, predictions: dict) -> pd.DataFrame:
    """One row per model: ranking (AUC, Gini) and whether the probability is honest.

    calibration_gap is the average distance between the predicted rate and the
    observed claim rate across ten bins. Smaller is better. Brier is the average
    squared error of the probabilities. Smaller is better.
    """
    y_true = np.asarray(y_true)
    rows = {}
    for name, y_pred in predictions.items():
        y_pred = np.asarray(y_pred, dtype=float)
        cal = calibration_table(y_true, y_pred)
        gap = np.average(np.abs(cal["predicted_rate"] - cal["actual_rate"]), weights=cal["n"])
        auc = roc_auc_score(y_true, y_pred)
        rows[name] = {
            "AUC": auc,
            "Gini": 2 * auc - 1,
            "Brier": brier_score_loss(y_true, y_pred),
            "calibration_gap": gap,
            "mean_predicted": y_pred.mean(),
        }
    return pd.DataFrame.from_dict(rows, orient="index")


def choose_model(scores: pd.DataFrame) -> str:
    """Choose the model for the app from a comparison_table.

    Highest AUC on the 2024 test wins. A simpler model takes the win when it is
    within 0.005 AUC and its calibration gap is within 0.002 of the best close
    model. That gap is too small to give up a model you can explain.
    """
    simpler_first = ["glm", "logistic", "tree", "forest", "xgboost", "lgbm"]
    best_auc = scores["AUC"].max()
    close = scores.loc[scores["AUC"] >= best_auc - 0.005]
    best_gap = close["calibration_gap"].min()
    candidates = close.loc[close["calibration_gap"] <= best_gap + 0.002]
    for name in simpler_first:
        if name in candidates.index:
            return name
    return scores["AUC"].idxmax()


def assign_risk_tier(prob, cutoffs: dict) -> np.ndarray:
    """Map probabilities to Low / Medium / High using cutoffs from config."""
    prob = np.asarray(prob)
    return np.where(prob < cutoffs["Low"], "Low",
                    np.where(prob < cutoffs["Medium"], "Medium", "High"))
