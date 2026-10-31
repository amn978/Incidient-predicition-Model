"""Prediction log + human feedback store (SQLite for the demo; swap for PostgreSQL in production).

This is the data source of the feedback loop: every prediction is logged, and the
human-confirmed/corrected severity is stored next to it for monitoring and retraining.
"""
from __future__ import annotations

import json
import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone

import pandas as pd

SCHEMA = """
CREATE TABLE IF NOT EXISTS predictions (
  id INTEGER PRIMARY KEY AUTOINCREMENT, ticket_id TEXT, ts TEXT, model_name TEXT, model_version TEXT,
  predicted TEXT, confidence REAL, route TEXT, probabilities TEXT, rules_fired TEXT, text TEXT);
CREATE TABLE IF NOT EXISTS feedback (
  id INTEGER PRIMARY KEY AUTOINCREMENT, ticket_id TEXT, ts TEXT, predicted TEXT, final_severity TEXT,
  reviewer TEXT, reason TEXT);
"""


@contextmanager
def _conn(path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    con = sqlite3.connect(path)
    try:
        con.executescript(SCHEMA)
        yield con
        con.commit()
    finally:
        con.close()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def log_prediction(path: str, rec: dict, text: str = "") -> None:
    with _conn(path) as c:
        c.execute("INSERT INTO predictions (ticket_id, ts, model_name, model_version, predicted, confidence, route,"
                  " probabilities, rules_fired, text) VALUES (?,?,?,?,?,?,?,?,?,?)",
                  (rec.get("ticket_id"), _now(), rec.get("model_name"), rec.get("model_version"), rec["severity"],
                   rec["confidence"], rec["route"], json.dumps(rec["probabilities"]),
                   json.dumps(rec["rules_fired"]), text[:500]))


def log_feedback(path: str, ticket_id: str, predicted: str, final_severity: str, reviewer: str = "", reason: str = "") -> None:
    with _conn(path) as c:
        c.execute("INSERT INTO feedback (ticket_id, ts, predicted, final_severity, reviewer, reason) VALUES (?,?,?,?,?,?)",
                  (ticket_id, _now(), predicted, final_severity, reviewer, reason))


def load_feedback(path: str) -> pd.DataFrame:
    with _conn(path) as c:
        return pd.read_sql_query("SELECT * FROM feedback ORDER BY id DESC", c)


def feedback_summary(path: str, classes: list[str], critical: str = "P1", low: tuple = ("P3", "P4")) -> dict:
    """Health metrics for the feedback loop (these drive retraining decisions)."""
    fb = load_feedback(path)
    if fb.empty:
        return {"n_feedback": 0}
    n = len(fb)
    rank = {c: i for i, c in enumerate(classes)}
    wrong = fb["predicted"] != fb["final_severity"]
    crit = fb[fb["final_severity"] == critical]
    miss = crit["predicted"].isin(low).sum() if len(crit) else 0
    under = (fb["predicted"].map(rank) > fb["final_severity"].map(rank)).mean()
    return {"n_feedback": n, "override_rate": float(wrong.mean()), "under_triage_rate": float(under),
            "critical_seen": int(len(crit)), "critical_missed": int(miss),
            "critical_miss_rate": float(miss / len(crit)) if len(crit) else None}
