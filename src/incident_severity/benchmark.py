"""Train + compare algorithms against the POC success metrics, then persist the best.

Pipeline per model:
  fit on TRAIN  ->  probabilities on VAL/TEST
  stage 1: argmax
  stage 2: + safety thresholds (fitted on VAL only)
  stage 3: + business rules   == "final" decision used for scoring and deployment
Best model is chosen on VALIDATION metrics only; TEST is reported, never used to choose.
"""
from __future__ import annotations

import json
import time
import warnings
from datetime import datetime, timezone

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.utils.class_weight import compute_sample_weight

from . import __version__
from .config import class_names, critical_idx, load_config, low_idx
from .data import prepare_datasets
from .evaluate import compute_metrics, confusion, routing_stats
from .models import REGISTRY, available_models
from .rules import RuleEngine
from .thresholds import DecisionPolicy, fit_policy, tradeoff_curve

warnings.filterwarnings("ignore")


def _log(msg: str, verbose: bool):
    if verbose:
        print(msg, flush=True)


def _latency_ms(model, policy: DecisionPolicy, rules: RuleEngine, X: pd.DataFrame, n: int = 60) -> dict:
    """End-to-end single-ticket latency: features + model + thresholds + rules."""
    rng = np.random.default_rng(0)
    idx = rng.choice(len(X), size=min(n, len(X)), replace=False)
    model.predict_proba(X.iloc[[0]])  # warm-up
    times = []
    for i in idx:
        row = X.iloc[[i]]
        t = time.perf_counter()
        proba = model.predict_proba(row)
        pred = policy.apply(proba)
        rules.apply(row, pred)
        times.append((time.perf_counter() - t) * 1000)
    return {"latency_ms_p50": float(np.median(times)), "latency_ms_p95": float(np.percentile(times, 95))}


def run_benchmark(cfg: dict | None = None, model_names: list[str] | None = None,
                  include_embeddings: bool = False, verbose: bool = True) -> dict:
    cfg = cfg or load_config()
    seed = cfg["project"]["random_state"]
    classes, c_idx, l_idx = class_names(cfg), critical_idx(cfg), low_idx(cfg)
    sm = cfg["success_metrics"]
    rules = RuleEngine(cfg)

    data = prepare_datasets(cfg)
    _log("\nData split (time-based, newest tickets held out for test):", verbose)
    _log(data.summary().to_string(index=False), verbose)

    names = model_names or available_models(include_optional=include_embeddings or cfg["models"]["include_embeddings"])
    rows, store = [], {}

    for name in names:
        spec = REGISTRY[name]
        _log(f"\n>>> Training {spec.label} ...", verbose)
        model = spec.build(seed)
        fit_kwargs = {}
        if spec.needs_sample_weight:
            fit_kwargs["clf__sample_weight"] = compute_sample_weight("balanced", data.y_train)
        t0 = time.perf_counter()
        model.fit(data.X_train, data.y_train, **fit_kwargs)
        train_s = time.perf_counter() - t0

        P_val, P_test = model.predict_proba(data.X_val), model.predict_proba(data.X_test)

        argmax_test = compute_metrics(data.y_test, P_test.argmax(1), classes, c_idx, l_idx, P_test)
        if cfg["thresholds"]["enabled"]:
            policy, info = fit_policy(P_val, data.y_val, classes, c_idx, l_idx,
                                      sm["critical_miss_rate_max"], cfg["thresholds"]["safety_margin"])
        else:
            policy, info = DecisionPolicy(c_idx, 1), {"feasible": True}

        val_pred, _ = rules.apply(data.X_val, policy.apply(P_val))
        test_pred, _ = rules.apply(data.X_test, policy.apply(P_test))
        final_val = compute_metrics(data.y_val, val_pred, classes, c_idx, l_idx)
        final_test = compute_metrics(data.y_test, test_pred, classes, c_idx, l_idx, P_test)
        route = routing_stats(P_test, test_pred, data.y_test, cfg["routing"]["review_confidence"])
        lat = _latency_ms(model, policy, rules, data.X_test)

        row = {
            "model": name, "label": spec.label, "family": spec.family, "train_seconds": round(train_s, 1),
            **lat,
            "argmax_macro_f1": argmax_test["macro_f1"], "argmax_critical_miss": argmax_test["critical_miss_rate"],
            "final_macro_f1": final_test["macro_f1"], "final_accuracy": final_test["accuracy"],
            "final_critical_miss": final_test["critical_miss_rate"],
            "final_critical_miss_ci_high": final_test["critical_miss_ci_high"],
            "final_critical_recall": final_test["critical_recall"],
            "final_critical_precision": final_test["critical_precision"],
            "ordinal_mae": final_test["ordinal_mae"], "log_loss": final_test["log_loss"],
            **{f"f1_{c}": final_test[f"f1_{c}"] for c in classes},
            **route,
            "val_macro_f1": final_val["macro_f1"], "val_critical_miss": final_val["critical_miss_rate"],
            "t_critical": policy.t_critical, "t_high": policy.t_high,
        }
        row["meets_f1"] = row["final_macro_f1"] >= sm["macro_f1_min"]
        row["meets_critical_miss"] = row["final_critical_miss"] <= sm["critical_miss_rate_max"]
        row["meets_latency"] = row["latency_ms_p95"] / 1000 <= sm["triage_seconds_max"]
        row["meets_all"] = bool(row["meets_f1"] and row["meets_critical_miss"] and row["meets_latency"])
        rows.append(row)
        store[name] = {"model": model, "policy": policy, "P_val": P_val, "P_test": P_test, "test_pred": test_pred}
        _log(f"    macro-F1 argmax {row['argmax_macro_f1']:.3f} -> final {row['final_macro_f1']:.3f} | "
             f"P1 miss {row['argmax_critical_miss']:.1%} -> {row['final_critical_miss']:.1%} | "
             f"p95 latency {row['latency_ms_p95']:.0f} ms | train {train_s:.0f}s", verbose)

    results = pd.DataFrame(rows)

    # ---- model selection on VALIDATION only: must satisfy miss target, then best macro-F1 ----
    limit = sm["critical_miss_rate_max"]
    results["val_feasible"] = results["val_critical_miss"] <= limit
    ranked = results.sort_values(["val_feasible", "val_macro_f1", "latency_ms_p95"], ascending=[False, False, True])
    best_name = ranked.iloc[0]["model"]
    results["selected"] = results["model"] == best_name
    results = results.sort_values("final_macro_f1", ascending=False).reset_index(drop=True)

    artifacts = _persist(cfg, data, results, store, best_name, classes, c_idx, l_idx)
    _log(f"\nBest model (chosen on validation): {best_name}", verbose)
    return {"results": results, "best": best_name, "data": data, "artifacts": artifacts}


def _persist(cfg, data, results, store, best_name, classes, c_idx, l_idx):
    paths, sm = cfg["paths"], cfg["success_metrics"]
    import os
    for d in (paths["models_dir"], paths["reports_dir"], paths["processed_dir"]):
        os.makedirs(d, exist_ok=True)

    best_row = results[results["model"] == best_name].iloc[0].to_dict()
    best = store[best_name]
    meta = {
        "model_name": best_name, "model_label": REGISTRY[best_name].label, "version": __version__,
        "trained_at": datetime.now(timezone.utc).isoformat(), "classes": classes,
        "policy": best["policy"].to_dict(),
        "train_rows": int(len(data.y_train)),
        "test_metrics": {k: (float(v) if isinstance(v, (int, float, np.floating)) else v)
                         for k, v in best_row.items() if k.startswith(("final_", "latency", "auto_", "ordinal", "f1_"))},
    }
    joblib.dump({"model": best["model"], "policy": best["policy"], "classes": classes, "meta": meta},
                os.path.join(paths["models_dir"], "best_model.joblib"))
    with open(os.path.join(paths["models_dir"], "model_card.json"), "w") as fh:
        json.dump(meta, fh, indent=2, default=str)

    results.to_csv(os.path.join(paths["reports_dir"], "benchmark_results.csv"), index=False)
    data.X_test.assign(severity=[classes[i] for i in data.y_test]).to_csv(
        os.path.join(paths["processed_dir"], "test_tickets.csv"), index=False)

    _charts(cfg, results, store, data, best_name, classes, c_idx, l_idx)
    _markdown(cfg, results, best_name)
    return {"model": os.path.join(paths["models_dir"], "best_model.joblib"),
            "report_csv": os.path.join(paths["reports_dir"], "benchmark_results.csv")}


def _charts(cfg, results, store, data, best_name, classes, c_idx, l_idx):
    rd, sm = cfg["paths"]["reports_dir"], cfg["success_metrics"]
    import os
    df = results.sort_values("final_macro_f1")
    colors = ["#2a9d8f" if s else "#8d99ae" for s in df["selected"]]

    fig, ax = plt.subplots(1, 3, figsize=(16, 5))
    b0 = ax[0].barh(df["label"], df["final_macro_f1"], color=colors)
    ax[0].bar_label(b0, fmt="%.3f", padding=3)
    ax[0].axvline(sm["macro_f1_min"], color="#d62828", ls="--", label=f"target {sm['macro_f1_min']:.0%}")
    ax[0].set_title("Macro F1 (test, higher is better)"); ax[0].set_xlim(0, 1); ax[0].legend(loc="lower right")
    b1 = ax[1].barh(df["label"], df["final_critical_miss"] * 100, color=colors)
    ax[1].bar_label(b1, fmt="%.1f%%", padding=3)
    ax[1].set_xlim(0, max(2.6, df["final_critical_miss"].max() * 100 * 1.3))
    ax[1].axvline(sm["critical_miss_rate_max"] * 100, color="#d62828", ls="--", label=f"target {sm['critical_miss_rate_max']:.0%}")
    ax[1].set_title("P1 missed as P3/P4 (%, test, lower is better)"); ax[1].set_yticklabels([]); ax[1].legend()
    b2 = ax[2].barh(df["label"], df["latency_ms_p95"], color=colors)
    ax[2].bar_label(b2, fmt="%.0f ms", padding=3)
    ax[2].set_xlim(0, df["latency_ms_p95"].max() * 1.25)
    ax[2].set_title("p95 latency per ticket (ms)"); ax[2].set_yticklabels([])
    ax[2].text(0.98, 0.04, "limit = 10,000 ms", transform=ax[2].transAxes, ha="right", color="#d62828")
    fig.suptitle("Algorithm comparison (green = selected on validation)", fontsize=13)
    fig.tight_layout(); fig.savefig(os.path.join(rd, "benchmark_comparison.png"), dpi=130); plt.close(fig)

    cm = confusion(data.y_test, store[best_name]["test_pred"], classes)
    fig, ax = plt.subplots(figsize=(5.5, 4.6))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(classes)), classes); ax.set_yticks(range(len(classes)), classes)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual"); ax.set_title(f"Confusion matrix: {REGISTRY[best_name].label}")
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, cm[i, j], ha="center", va="center", color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.colorbar(im, fraction=0.046); fig.tight_layout()
    fig.savefig(os.path.join(rd, "confusion_matrix_best.png"), dpi=130); plt.close(fig)

    pol = store[best_name]["policy"]
    curve = tradeoff_curve(store[best_name]["P_test"], data.y_test, classes, c_idx, l_idx, pol.t_high)
    t, miss, f1, _ = zip(*curve)
    fig, ax = plt.subplots(figsize=(6.5, 4.4))
    ax.plot(t, np.array(miss) * 100, color="#d62828", marker="o", label="P1 missed as P3/P4 (%)")
    ax.set_xlabel("P1 probability threshold (lower = more cautious)"); ax.set_ylabel("P1 miss rate (%)", color="#d62828")
    ax2 = ax.twinx(); ax2.plot(t, f1, color="#2a9d8f", marker="s", label="Macro F1"); ax2.set_ylabel("Macro F1", color="#2a9d8f")
    ax.axhline(sm["critical_miss_rate_max"] * 100, color="grey", ls="--")
    ax.set_title("Safety vs accuracy trade-off (best model, test)"); fig.tight_layout()
    fig.savefig(os.path.join(rd, "threshold_tradeoff.png"), dpi=130); plt.close(fig)


def _markdown(cfg, results, best_name):
    sm = cfg["success_metrics"]
    import os
    lines = ["# Benchmark results (test set = newest tickets)", "",
             f"Targets: macro F1 >= {sm['macro_f1_min']:.0%}, P1 missed as P3/P4 <= {sm['critical_miss_rate_max']:.0%}, "
             f"latency <= {sm['triage_seconds_max']} s.", "",
             "| Model | Macro F1 | P1 miss % | P1 recall | p95 ms | Train s | Meets all |", "|---|---|---|---|---|---|---|"]
    for _, r in results.iterrows():
        star = " (selected)" if r["model"] == best_name else ""
        lines.append(f"| {r['label']}{star} | {r['final_macro_f1']:.3f} | {r['final_critical_miss']:.1%} | "
                     f"{r['final_critical_recall']:.1%} | {r['latency_ms_p95']:.0f} | {r['train_seconds']:.0f} | "
                     f"{'yes' if r['meets_all'] else 'no'} |")
    with open(os.path.join(cfg["paths"]["reports_dir"], "benchmark_summary.md"), "w") as fh:
        fh.write("\n".join(lines) + "\n")
