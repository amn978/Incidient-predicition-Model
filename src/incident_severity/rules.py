"""Business rule overlay. Config-driven, can only RAISE severity (never lower)."""
from __future__ import annotations

import re

import numpy as np
import pandas as pd


class RuleEngine:
    def __init__(self, cfg: dict):
        rcfg = cfg.get("rules", {})
        self.enabled = bool(rcfg.get("enabled", False))
        classes = list(cfg["classes"])
        self.rules = []
        for item in rcfg.get("items", []):
            self.rules.append({
                "name": item["name"],
                "re": re.compile(item["pattern"], re.IGNORECASE),
                "floor": classes.index(item["min_severity"]),
                "envs": set(item.get("environments", [])),
            })

    def apply(self, df: pd.DataFrame, pred: np.ndarray):
        """Returns (new_pred, fired) where fired[i] is the list of rule names that changed or supported row i."""
        pred = np.asarray(pred).copy()
        fired: list[list[str]] = [[] for _ in range(len(pred))]
        if not self.enabled or not self.rules:
            return pred, fired
        text = (df["short_description"].fillna("") + " " + df["description"].fillna("")).tolist()
        env = df["environment"].fillna("").tolist()
        for i in range(len(pred)):
            for r in self.rules:
                if env[i] in r["envs"] and r["re"].search(text[i]):
                    fired[i].append(r["name"])
                    pred[i] = min(pred[i], r["floor"])
        return pred, fired
