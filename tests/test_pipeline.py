"""End-to-end: train two fast models on a small dataset, serve predictions, test the API."""
import os
import copy

import pytest

from incident_severity.config import load_config
from incident_severity.data_generator import generate_tickets


@pytest.fixture(scope="module")
def tiny_cfg(tmp_path_factory):
    root = tmp_path_factory.mktemp("proj")
    cfg = load_config()
    cfg = copy.deepcopy(cfg)
    for k, sub in dict(raw_tickets="tickets.csv", cmdb="cmdb.csv", feedback_db="fb.db").items():
        cfg["paths"][k] = str(root / sub)
    for k in ("models_dir", "reports_dir", "processed_dir", "data_dir"):
        cfg["paths"][k] = str(root / k)
    df, cmdb = generate_tickets(2500, seed=5)
    df.to_csv(cfg["paths"]["raw_tickets"], index=False); cmdb.to_csv(cfg["paths"]["cmdb"], index=False)
    return cfg


def test_benchmark_and_predictor(tiny_cfg):
    from incident_severity.benchmark import run_benchmark
    from incident_severity.predict import SeverityPredictor
    out = run_benchmark(tiny_cfg, ["logreg", "complement_nb"], verbose=False)
    assert out["best"] in {"logreg", "complement_nb"}
    assert os.path.exists(out["artifacts"]["model"])
    assert (out["results"]["final_macro_f1"] > 0.4).all()

    p = SeverityPredictor(tiny_cfg)
    res = p.predict([
        {"short_description": "Primary database is down for checkout-service", "environment": "prod", "affected_service": "checkout-service"},
        {"short_description": "Resource tagging violations in hr-portal account", "environment": "dev", "affected_service": "hr-portal"},
    ])
    assert res[0]["severity"] == "P1" and res[1]["severity"] in {"P3", "P4"}
    assert set(res[0]["probabilities"]) == {"P1", "P2", "P3", "P4"}
    assert res[0]["latency_ms"] < 10_000


def test_api_roundtrip(tiny_cfg, monkeypatch, tmp_path):
    pytest.importorskip("fastapi")
    import yaml
    from fastapi.testclient import TestClient

    # write a config whose (absolute) paths point at the temp project built above
    raw = {k: v for k, v in tiny_cfg.items() if k != "_root"}
    cfg_dir = tmp_path / "config"
    cfg_dir.mkdir()
    with open(cfg_dir / "config.yaml", "w") as fh:
        yaml.safe_dump(raw, fh)
    monkeypatch.setenv("ISP_CONFIG", str(cfg_dir / "config.yaml"))

    from incident_severity.api import app
    with TestClient(app) as client:
        assert client.get("/health").json()["status"] == "ok"
        r = client.post("/predict", json={"short_description": "Primary database is down for checkout-service",
                                          "environment": "prod", "affected_service": "checkout-service"})
        assert r.status_code == 200 and r.json()["severity"] == "P1"
        assert client.post("/predict", json={"short_description": "x"}).status_code == 422
        fb = client.post("/feedback", json={"ticket_id": "T1", "predicted": "P1", "final_severity": "P2"})
        assert fb.status_code == 200
        assert client.get("/feedback/summary").json()["n_feedback"] == 1
