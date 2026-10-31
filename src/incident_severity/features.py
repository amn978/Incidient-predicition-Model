"""Stateless feature building: raw ticket dataframe -> model-ready dataframe.

Only information available AT TICKET CREATION is used. Text is cleaned and PII
(emails, IPs, ids) is masked before it reaches any model.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

CAT_COLS = ["category", "subcategory", "environment", "channel", "affected_service",
            "service_tier", "customer_facing", "users_bucket", "hour_bucket", "is_weekend"]
NUM_COLS = ["users_log", "hour", "weekday"]

_RE_EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
_RE_IP = re.compile(r"\b\d{1,3}(?:\.\d{1,3}){3}\b")
_RE_TICKET = re.compile(r"\b(?:inc|chg|prb)\d+\b")
_RE_HEX = re.compile(r"\b[0-9a-f]{8,}\b")
_RE_NUM = re.compile(r"\b\d+(?:\.\d+)?\b")
_RE_WS = re.compile(r"\s+")


def clean_text(text: str) -> str:
    """Lowercase, mask PII / identifiers, collapse whitespace."""
    t = str(text).lower()
    t = _RE_EMAIL.sub(" emailaddr ", t)
    t = _RE_IP.sub(" ipaddr ", t)
    t = _RE_TICKET.sub(" ticketref ", t)
    t = _RE_HEX.sub(" hexid ", t)
    t = _RE_NUM.sub(" num ", t)
    return _RE_WS.sub(" ", t).strip()


def _hour_bucket(h: pd.Series) -> pd.Series:
    return pd.cut(h, bins=[-1, 5, 11, 17, 24], labels=["night", "morning", "afternoon", "evening"]).astype(str)


class FeatureBuilder(BaseEstimator, TransformerMixin):
    """Turns the raw ticket frame into the columns the model expects."""

    def fit(self, X, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X
        out = pd.DataFrame(index=df.index)
        text = df["short_description"].fillna("").astype(str) + " . " + df["description"].fillna("").astype(str)
        out["text"] = text.map(clean_text)

        out["category"] = df["category"].fillna("unknown").replace("", "unknown").astype(str)
        out["subcategory"] = df["subcategory"].fillna("unknown").replace("", "unknown").astype(str)
        out["environment"] = df["environment"].fillna("unknown").astype(str)
        out["channel"] = df["channel"].fillna("unknown").astype(str)
        out["affected_service"] = df["affected_service"].fillna("unknown").astype(str)
        out["service_tier"] = df["service_tier"].fillna(-1).astype(int).astype(str)
        out["customer_facing"] = df["customer_facing"].fillna(-1).astype(int).astype(str)

        users = pd.to_numeric(df["users_affected"], errors="coerce")
        out["users_bucket"] = pd.cut(users, bins=[-1, 0, 10, 100, 1000, np.inf],
                                     labels=["0", "1-10", "11-100", "101-1000", "1000+"]).astype(object).fillna("unknown")
        out["users_log"] = np.log1p(users).fillna(-1.0)

        ts = pd.to_datetime(df["created_at"])
        out["hour"] = ts.dt.hour
        out["weekday"] = ts.dt.weekday
        out["hour_bucket"] = _hour_bucket(ts.dt.hour)
        out["is_weekend"] = (ts.dt.weekday >= 5).astype(int).astype(str)
        return out
