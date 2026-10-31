"""Metrics aligned to the POC success criteria."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, log_loss,
                             precision_recall_fscore_support)


def wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson interval for a proportion (honest uncertainty when P1 counts are small)."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    denom = 1 + z ** 2 / n
    centre = (p + z ** 2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def compute_metrics(y_true, y_pred, classes: list[str], critical_idx: int, low_idx: list[int],
                    proba=None) -> dict:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    k = len(classes)
    labels = list(range(k))
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)

    out = {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": float(np.mean(f1)),
        "weighted_f1": f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0),
        "ordinal_mae": float(np.abs(y_true - y_pred).mean()),
        "under_triage_rate": float((y_pred > y_true).mean()),   # predicted LESS severe than truth
        "over_triage_rate": float((y_pred < y_true).mean()),    # predicted MORE severe than truth
    }
    for i, c in enumerate(classes):
        out[f"f1_{c}"], out[f"recall_{c}"], out[f"precision_{c}"] = float(f1[i]), float(rec[i]), float(prec[i])

    crit = y_true == critical_idx
    n_crit = int(crit.sum())
    miss = int(np.isin(y_pred[crit], list(low_idx)).sum()) if n_crit else 0
    out["n_critical"] = n_crit
    out["critical_misses"] = miss
    out["critical_miss_rate"] = miss / n_crit if n_crit else float("nan")
    out["critical_miss_ci_high"] = wilson_interval(miss, n_crit)[1]
    out["critical_recall"] = float(rec[critical_idx])
    out["critical_precision"] = float(prec[critical_idx])
    if proba is not None:
        out["log_loss"] = float(log_loss(y_true, proba, labels=labels))
    return out


def confusion(y_true, y_pred, classes: list[str]):
    return confusion_matrix(y_true, y_pred, labels=list(range(len(classes))))


def routing_stats(proba, pred, y_true, review_confidence: float) -> dict:
    """How many tickets could be auto-assigned, and how accurate are those."""
    conf = proba[np.arange(len(pred)), pred]
    auto = conf >= review_confidence
    return {
        "auto_assign_rate": float(auto.mean()),
        "auto_accuracy": float((pred[auto] == y_true[auto]).mean()) if auto.any() else float("nan"),
    }
