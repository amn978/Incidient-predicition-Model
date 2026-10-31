"""Data loading, CMDB enrichment, leakage guard and time-based splitting."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class Splits:
    X_train: pd.DataFrame
    y_train: np.ndarray
    X_val: pd.DataFrame
    y_val: np.ndarray
    X_test: pd.DataFrame
    y_test: np.ndarray
    classes: list[str]

    def summary(self) -> pd.DataFrame:
        rows = []
        for name, y, X in [("train", self.y_train, self.X_train), ("val", self.y_val, self.X_val),
                           ("test", self.y_test, self.X_test)]:
            row = {"split": name, "rows": len(y),
                   "from": X["created_at"].min().date(), "to": X["created_at"].max().date()}
            for i, c in enumerate(self.classes):
                row[c] = int((y == i).sum())
            rows.append(row)
        return pd.DataFrame(rows)


def enrich_with_cmdb(tickets: pd.DataFrame, cmdb: pd.DataFrame) -> pd.DataFrame:
    """Join service criticality / customer-facing flag from the CMDB (idempotent)."""
    tickets = tickets.drop(columns=[c for c in ("service_tier", "customer_facing", "owner_team")
                                    if c in tickets.columns])
    out = tickets.merge(cmdb[["affected_service", "service_tier", "customer_facing"]],
                        on="affected_service", how="left")
    out["service_tier"] = out["service_tier"].fillna(-1).astype(int)
    out["customer_facing"] = out["customer_facing"].fillna(-1).astype(int)
    out.index = tickets.index
    return out


def drop_leakage(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Remove columns that are only known after triage/resolution."""
    cols = [c for c in cfg["data"]["leakage_columns"] if c in df.columns]
    return df.drop(columns=cols)


def time_split(df: pd.DataFrame, train_frac: float, val_frac: float):
    """Oldest tickets -> train, newest -> test. Mimics real deployment (no random shuffle)."""
    n = len(df)
    i1, i2 = int(n * train_frac), int(n * (train_frac + val_frac))
    return df.iloc[:i1], df.iloc[i1:i2], df.iloc[i2:]


def prepare_datasets(cfg: dict) -> Splits:
    tickets = pd.read_csv(cfg["paths"]["raw_tickets"], parse_dates=[cfg["data"]["time_col"]])
    cmdb = pd.read_csv(cfg["paths"]["cmdb"])
    tickets = tickets.sort_values(cfg["data"]["time_col"]).reset_index(drop=True)
    tickets = enrich_with_cmdb(tickets, cmdb)

    classes = list(cfg["classes"])
    target = cfg["data"]["target_col"]
    y_all = tickets[target].map({c: i for i, c in enumerate(classes)})
    if y_all.isna().any():
        raise ValueError(f"Unknown severity labels found: {tickets.loc[y_all.isna(), target].unique()}")

    X_all = drop_leakage(tickets.drop(columns=[target]), cfg)
    tr, va, te = time_split(X_all, cfg["split"]["train_frac"], cfg["split"]["val_frac"])
    y_all = y_all.to_numpy()
    n1, n2 = len(tr), len(tr) + len(va)
    return Splits(tr, y_all[:n1], va, y_all[n1:n2], te, y_all[n2:], classes)
