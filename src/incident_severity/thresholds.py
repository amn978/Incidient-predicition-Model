"""Decision policy: turn probabilities into severity with an asymmetric safety bias.

Plain argmax treats all mistakes equally. Missing a P1 is far worse than a false alarm,
so two tunable thresholds push uncertain cases UP the severity ladder:

  1. If P(P1) >= t_critical               -> predict P1
  2. If P(P1)+P(P2) >= t_high and the model said "lower than P2" -> predict P2

Both thresholds are fitted on the VALIDATION set only (never on test) to maximise
macro-F1 subject to the critical miss-rate target.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from .evaluate import compute_metrics

OFF = 1.01   # threshold value that disables a rule


@dataclass
class DecisionPolicy:
    critical_idx: int = 0
    high_idx: int = 1
    t_critical: float = OFF
    t_high: float = OFF

    def apply(self, proba: np.ndarray) -> np.ndarray:
        pred = proba.argmax(axis=1)
        pred = np.where(proba[:, self.critical_idx] >= self.t_critical, self.critical_idx, pred)
        sev_mass = proba[:, self.critical_idx] + proba[:, self.high_idx]
        pred = np.where((pred > self.high_idx) & (sev_mass >= self.t_high), self.high_idx, pred)
        return pred

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "DecisionPolicy":
        return cls(**d)


def fit_policy(proba, y, classes, critical_idx, low_idx, target_miss: float, safety_margin: float = 0.75):
    """Grid-search thresholds on validation data. Returns (policy, info)."""
    grid = list(np.round(np.arange(0.05, 0.96, 0.05), 2)) + [OFF]
    limit = target_miss * safety_margin
    best, fallback = None, None
    base = compute_metrics(y, proba.argmax(1), classes, critical_idx, low_idx)

    for t1 in grid:
        for t2 in grid:
            pol = DecisionPolicy(critical_idx, 1, t1, t2)
            m = compute_metrics(y, pol.apply(proba), classes, critical_idx, low_idx)
            key_ok = (m["macro_f1"], t1 + t2)                   # prefer higher F1, then less aggressive
            if m["critical_miss_rate"] <= limit and (best is None or key_ok > best[0]):
                best = (key_ok, pol, m)
            key_fb = (-m["critical_miss_rate"], m["macro_f1"])  # nothing feasible: minimise misses
            if fallback is None or key_fb > fallback[0]:
                fallback = (key_fb, pol, m)

    feasible = best is not None
    _, policy, m = best if feasible else fallback
    info = {"feasible": feasible, "val_macro_f1_before": base["macro_f1"],
            "val_miss_before": base["critical_miss_rate"], "val_macro_f1_after": m["macro_f1"],
            "val_miss_after": m["critical_miss_rate"], "limit_used": limit}
    return policy, info


def tradeoff_curve(proba, y, classes, critical_idx, low_idx, t_high: float):
    """Critical-miss vs macro-F1 as the P1 threshold varies (for the manager chart)."""
    rows = []
    for t1 in np.round(np.arange(0.05, 0.96, 0.05), 2):
        m = compute_metrics(y, DecisionPolicy(critical_idx, 1, t1, t_high).apply(proba), classes, critical_idx, low_idx)
        rows.append((float(t1), m["critical_miss_rate"], m["macro_f1"], m["critical_precision"]))
    return rows
