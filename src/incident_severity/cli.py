"""Command line entry point:  isp generate-data | benchmark | predict | info"""
from __future__ import annotations

import argparse
import json
import os

from .config import load_config


def cmd_generate(args):
    from .data_generator import generate_tickets
    cfg = load_config(args.config)
    d = cfg["data"]
    tickets, cmdb = generate_tickets(args.n or d["n_tickets"], d["start_date"], d["end_date"],
                                     d["label_noise"], cfg["project"]["random_state"], d.get("score_noise", 0.45))
    os.makedirs(cfg["paths"]["data_dir"], exist_ok=True)
    tickets.to_csv(cfg["paths"]["raw_tickets"], index=False)
    cmdb.to_csv(cfg["paths"]["cmdb"], index=False)
    print(f"Wrote {len(tickets)} tickets -> {cfg['paths']['raw_tickets']}")
    print(tickets["severity"].value_counts(normalize=True).sort_index().round(3).to_string())


def cmd_benchmark(args):
    from .benchmark import run_benchmark
    cfg = load_config(args.config)
    names = args.models.split(",") if args.models else None
    out = run_benchmark(cfg, names, include_embeddings=args.embeddings)
    cols = ["label", "final_macro_f1", "final_critical_miss", "final_critical_recall", "latency_ms_p95",
            "train_seconds", "meets_all", "selected"]
    print("\n" + out["results"][cols].round(3).to_string(index=False))
    print(f"\nSaved model -> {out['artifacts']['model']}\nReports  -> {cfg['paths']['reports_dir']}")


def cmd_predict(args):
    from .predict import SeverityPredictor
    p = SeverityPredictor(load_config(args.config))
    res = p.predict([{"short_description": args.text, "description": args.description or "",
                      "environment": args.env, "affected_service": args.service, "channel": "portal"}])[0]
    print(json.dumps(res, indent=2, default=str))


def cmd_info(args):
    path = os.path.join(load_config(args.config)["paths"]["models_dir"], "model_card.json")
    print(open(path).read())


def main(argv=None):
    ap = argparse.ArgumentParser(prog="isp", description="Incident severity prediction toolkit")
    ap.add_argument("--config", default=None, help="path to config.yaml")
    sub = ap.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("generate-data", help="create synthetic demo tickets + CMDB")
    g.add_argument("--n", type=int, default=None); g.set_defaults(fn=cmd_generate)

    b = sub.add_parser("benchmark", help="train and compare all algorithms, save the best")
    b.add_argument("--models", default=None, help="comma list, e.g. logreg,lightgbm")
    b.add_argument("--embeddings", action="store_true", help="include sentence-embedding model")
    b.set_defaults(fn=cmd_benchmark)

    p = sub.add_parser("predict", help="score a single ticket from the terminal")
    p.add_argument("text"); p.add_argument("--description", default="")
    p.add_argument("--env", default="prod"); p.add_argument("--service", default="checkout-service")
    p.set_defaults(fn=cmd_predict)

    i = sub.add_parser("info", help="print the saved model card"); i.set_defaults(fn=cmd_info)
    args = ap.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
