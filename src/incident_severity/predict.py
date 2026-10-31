"""Inference service logic: raw ticket(s) -> severity, confidence, routing, explanation hints."""
from __future__ import annotations

import os
import time

import joblib
import numpy as np
import pandas as pd

from .config import load_config
from .data import enrich_with_cmdb
from .rules import RuleEngine

ACTIONS = {
    "P1": "Declare major incident: page on-call and incident commander, open war room.",
    "P2": "Page the owning on-call team; update stakeholders every 1-2 hours.",
    "P3": "Assign to the owning team queue; handle within business hours.",
    "P4": "Add to backlog; handle on the next business day.",
}
DEFAULTS = {"ticket_id": None, "created_at": None, "channel": "portal", "category": None, "subcategory": None,
            "affected_service": None, "environment": "prod", "short_description": "", "description": "",
            "users_affected": np.nan}
RULE_CONFIDENCE = 0.95   # a matched business rule is deterministic policy, so confidence is high


class SeverityPredictor:
    def __init__(self, cfg: dict | None = None, artifact_path: str | None = None):
        self.cfg = cfg or load_config()
        path = artifact_path or os.path.join(self.cfg["paths"]["models_dir"], "best_model.joblib")
        if not os.path.exists(path):
            raise FileNotFoundError(f"No trained model at {path}. Run: isp benchmark")
        art = joblib.load(path)
        self.model, self.policy, self.classes, self.meta = art["model"], art["policy"], art["classes"], art["meta"]
        self.cmdb = pd.read_csv(self.cfg["paths"]["cmdb"])
        self.rules = RuleEngine(self.cfg)
        self.review_conf = self.cfg["routing"]["review_confidence"]

    def _frame(self, tickets: list[dict]) -> pd.DataFrame:
        df = pd.DataFrame([{**DEFAULTS, **t} for t in tickets])
        df["created_at"] = pd.to_datetime(df["created_at"].fillna(pd.Timestamp.now()))
        df["users_affected"] = pd.to_numeric(df["users_affected"], errors="coerce")
        df["environment"] = df["environment"].fillna("prod")
        return enrich_with_cmdb(df, self.cmdb)

    def predict(self, tickets: list[dict]) -> list[dict]:
        t0 = time.perf_counter()
        df = self._frame(tickets)
        proba = self.model.predict_proba(df)
        base = proba.argmax(1)
        pred = self.policy.apply(proba)
        final, fired = self.rules.apply(df, pred)
        ms = (time.perf_counter() - t0) * 1000 / max(len(df), 1)

        out = []
        for i in range(len(df)):
            conf = float(proba[i, final[i]])
            if fired[i] and final[i] != pred[i]:
                conf = max(conf, RULE_CONFIDENCE)
            sev = self.classes[final[i]]
            out.append({
                "ticket_id": tickets[i].get("ticket_id"),
                "severity": sev,
                "confidence": round(conf, 4),
                "probabilities": {c: round(float(proba[i, j]), 4) for j, c in enumerate(self.classes)},
                "route": "auto_assign" if conf >= self.review_conf else "human_review",
                "recommended_action": ACTIONS[sev],
                "model_argmax": self.classes[base[i]],
                "escalated_by_threshold": bool(pred[i] != base[i]),
                "rules_fired": fired[i],
                "model_name": self.meta["model_name"],
                "model_version": self.meta["version"],
                "latency_ms": round(ms, 1),
            })
        return out
