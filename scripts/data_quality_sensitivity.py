"""How much does data quality limit the achievable F1?

Re-generates the demo data with different amounts of 'unobserved context' (score_noise) and
trains the fast baseline each time. Shows the manager that the 80% target is mostly a DATA
question (are impact, users affected, service criticality captured at ticket creation?),
not an algorithm question.

    python scripts/data_quality_sensitivity.py
"""
import copy
import os
import tempfile

import pandas as pd

from incident_severity.benchmark import run_benchmark
from incident_severity.config import load_config
from incident_severity.data_generator import generate_tickets

LEVELS = [("rich context captured", 0.25), ("moderate (demo default)", 0.45), ("poor context captured", 0.80)]


def main():
    base = load_config()
    rows = []
    for label, noise in LEVELS:
        cfg = copy.deepcopy(base)
        tmp = tempfile.mkdtemp()
        for k in ("models_dir", "reports_dir", "processed_dir", "data_dir"):
            cfg["paths"][k] = os.path.join(tmp, k)
        cfg["paths"]["raw_tickets"], cfg["paths"]["cmdb"] = os.path.join(tmp, "t.csv"), os.path.join(tmp, "c.csv")
        d = cfg["data"]
        t, c = generate_tickets(d["n_tickets"], d["start_date"], d["end_date"], d["label_noise"],
                                cfg["project"]["random_state"], noise)
        t.to_csv(cfg["paths"]["raw_tickets"], index=False); c.to_csv(cfg["paths"]["cmdb"], index=False)
        r = run_benchmark(cfg, ["logreg", "lightgbm"], verbose=False)["results"].iloc[0]
        rows.append({"data quality": label, "score_noise": noise, "best model": r["label"],
                     "macro F1": round(r["final_macro_f1"], 3), "P1 miss %": round(r["final_critical_miss"] * 100, 1),
                     "P1 recall %": round(r["final_critical_recall"] * 100, 1)})
        print(rows[-1], flush=True)
    out = pd.DataFrame(rows)
    os.makedirs(base["paths"]["reports_dir"], exist_ok=True)
    out.to_csv(os.path.join(base["paths"]["reports_dir"], "data_quality_sensitivity.csv"), index=False)
    print("\n", out.to_string(index=False))


if __name__ == "__main__":
    main()
