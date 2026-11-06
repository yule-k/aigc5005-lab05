"""Metrics, the cost of a set of decisions and the threshold with the lowest cost."""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

VOUCHER = 4  # every flagged order costs a $4 voucher
MISSED_LATE = 15  # every late order we did not flag costs $15
THRESHOLDS = np.round(np.arange(0.05, 0.96, 0.05), 2)  # 0.05, 0.10, ..., 0.95


def order_cost(y_true, flagged) -> int:
    """$4 for every order flagged, plus $15 for every late order not flagged."""
    y_true = np.asarray(y_true)
    flagged = np.asarray(flagged, dtype=bool)
    missed = (y_true == 1) & ~flagged
    return int(VOUCHER * flagged.sum() + MISSED_LATE * missed.sum())


def cost_by_threshold(y_true, probability) -> pd.DataFrame:
    """Total cost at every threshold from 0.05 to 0.95."""
    probability = np.asarray(probability)
    rows = [
        {"threshold": t, "cost": order_cost(y_true, probability >= t)}
        for t in THRESHOLDS
    ]
    return pd.DataFrame(rows)


def choose_threshold(y_true, probability) -> float:
    """The threshold with the lowest total cost. On a tie, the lowest threshold."""
    costs = cost_by_threshold(y_true, probability)
    return float(costs.loc[costs["cost"].idxmin(), "threshold"])


def evaluate(y_true, probability, threshold: float) -> dict:
    """Confusion matrix, precision, recall, F1 and ROC-AUC at the given threshold."""
    flagged = np.asarray(probability) >= threshold
    return {
        "threshold": threshold,
        "confusion_matrix": confusion_matrix(y_true, flagged).tolist(),
        "precision": round(precision_score(y_true, flagged, zero_division=0), 3),
        "recall": round(recall_score(y_true, flagged), 3),
        "f1": round(f1_score(y_true, flagged), 3),
        "roc_auc": round(roc_auc_score(y_true, probability), 3),
    }
