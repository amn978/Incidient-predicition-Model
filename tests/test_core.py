"""Fast unit tests (no trained model needed)."""
import numpy as np
import pandas as pd

from incident_severity.config import load_config
from incident_severity.data_generator import generate_tickets
from incident_severity.data import drop_leakage, enrich_with_cmdb, time_split
from incident_severity.evaluate import compute_metrics, wilson_interval
from incident_severity.features import FeatureBuilder, clean_text
from incident_severity.rules import RuleEngine
from incident_severity.thresholds import DecisionPolicy, fit_policy

CLASSES = ["P1", "P2", "P3", "P4"]


def test_generator_shape_and_classes():
    df, cmdb = generate_tickets(1500, seed=1)
    assert set(df["severity"]) == set(CLASSES)
    assert df["ticket_id"].is_unique
    assert {"affected_service", "service_tier", "customer_facing"} <= set(cmdb.columns)


def test_leakage_columns_are_dropped():
    cfg = load_config()
    df, _ = generate_tickets(300, seed=2)
    cleaned = drop_leakage(df, cfg)
    for col in cfg["data"]["leakage_columns"]:
        assert col not in cleaned.columns


def test_time_split_is_chronological_and_disjoint():
    df, cmdb = generate_tickets(1000, seed=3)
    df = enrich_with_cmdb(df.sort_values("created_at").reset_index(drop=True), cmdb)
    tr, va, te = time_split(df, 0.6, 0.2)
    assert tr["created_at"].max() <= va["created_at"].min() <= va["created_at"].max() <= te["created_at"].min()
    assert len(tr) + len(va) + len(te) == len(df)


def test_clean_text_masks_pii():
    t = clean_text("Contact john.doe@corp.com from 10.1.2.3 about INC123456 id deadbeef1234")
    assert "john" not in t and "10.1.2.3" not in t and "inc123456" not in t and "deadbeef1234" not in t


def test_feature_builder_columns():
    cfg = load_config()
    df, cmdb = generate_tickets(50, seed=4)
    df = enrich_with_cmdb(df, cmdb)
    out = FeatureBuilder().transform(df)
    for col in ["text", "category", "users_bucket", "users_log", "hour", "hour_bucket", "is_weekend"]:
        assert col in out.columns
    assert out["text"].notna().all()


def test_critical_miss_rate_definition():
    y_true = np.array([0, 0, 0, 0, 1, 2, 3])
    y_pred = np.array([0, 1, 2, 3, 1, 2, 3])        # of 4 P1s: one -> P2 (not a "miss"), two -> P3/P4
    m = compute_metrics(y_true, y_pred, CLASSES, 0, [2, 3])
    assert m["n_critical"] == 4 and m["critical_misses"] == 2
    assert abs(m["critical_miss_rate"] - 0.5) < 1e-9
    assert abs(m["critical_recall"] - 0.25) < 1e-9


def test_wilson_interval_contains_estimate():
    lo, hi = wilson_interval(2, 100)
    assert lo < 0.02 < hi


def test_decision_policy_escalates_uncertain_critical():
    proba = np.array([[0.30, 0.20, 0.25, 0.25],     # argmax=P1 anyway
                      [0.28, 0.12, 0.30, 0.30],     # argmax P3, but P1 prob 0.28 -> escalate with t1=0.25
                      [0.05, 0.05, 0.10, 0.80]])    # clear P4 stays
    pol = DecisionPolicy(0, 1, t_critical=0.25, t_high=1.01)
    assert list(pol.apply(proba)) == [0, 0, 3]


def test_fit_policy_reduces_misses_when_possible():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 4, 800)
    logits = rng.normal(0, 1, (800, 4)); logits[np.arange(800), y] += 1.6
    proba = np.exp(logits) / np.exp(logits).sum(1, keepdims=True)
    base = compute_metrics(y, proba.argmax(1), CLASSES, 0, [2, 3])["critical_miss_rate"]
    pol, info = fit_policy(proba, y, CLASSES, 0, [2, 3], target_miss=0.02, safety_margin=0.75)
    assert info["val_miss_after"] <= base + 1e-9


def test_rules_only_raise_severity_and_respect_environment():
    cfg = load_config()
    eng = RuleEngine(cfg)
    df = pd.DataFrame({"short_description": ["Ransomware detected on host", "Primary database is down for x", "Primary database is down for x"],
                       "description": ["", "", ""], "environment": ["dev", "prod", "dev"]})
    pred, fired = eng.apply(df, np.array([3, 3, 3]))
    assert list(pred) == [0, 0, 3]                 # ransomware any env, DB-down only in prod
    assert fired[0] == ["ransomware"] and fired[2] == []
    pred2, _ = eng.apply(df, np.array([0, 0, 0]))
    assert list(pred2) == [0, 0, 0]                # never lowers
