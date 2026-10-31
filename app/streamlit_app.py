"""Manager demo UI.   Run:  streamlit run app/streamlit_app.py"""
from __future__ import annotations

import os
import time

import numpy as np
import pandas as pd
import streamlit as st

from incident_severity.config import load_config
from incident_severity.predict import SeverityPredictor
from incident_severity.storage import feedback_summary, load_feedback, log_feedback, log_prediction

st.set_page_config(page_title="Incident Severity AI", page_icon="🚨", layout="wide")
BADGE = {"P1": "🔴 P1 Critical", "P2": "🟠 P2 High", "P3": "🟡 P3 Medium", "P4": "🟢 P4 Low"}

EXAMPLES = {
    "Prod primary DB down (expect P1)": dict(short_description="Primary database is down for checkout-service",
        description="Requests are failing or timing out. Environment: production. Customers are unable to complete transactions.",
        environment="prod", affected_service="checkout-service", channel="monitoring", users_affected=5000),
    "Leaked access keys in prod (expect P1 via rule)": dict(short_description="AWS access key found in public GitHub repo for payments-api",
        description="Flagged by the security tooling and needs review. Observed in prod.",
        environment="prod", affected_service="payments-api", channel="portal"),
    "5xx spike on customer portal (expect P2)": dict(short_description="Spike of 502/503 errors on load balancer for customer-portal",
        description="Some users are seeing intermittent errors. Observed in prod.", environment="prod",
        affected_service="customer-portal", channel="monitoring", users_affected=300),
    "Pipeline failure in dev (expect P4)": dict(short_description="CI pipeline failing for internal-wiki",
        description="The job or process did not complete successfully. Only reproducible in dev. No user impact at this time.",
        environment="dev", affected_service="internal-wiki", channel="portal"),
    "Tagging violations (expect P4)": dict(short_description="Resource tagging violations in hr-portal account",
        description="Found during routine review. No immediate service impact observed.", environment="staging",
        affected_service="hr-portal", channel="email"),
}


@st.cache_resource
def get_predictor():
    cfg = load_config()
    return cfg, SeverityPredictor(cfg)


try:
    cfg, predictor = get_predictor()
except FileNotFoundError:
    st.error("No trained model found. Run `isp generate-data` then `isp benchmark` first.")
    st.stop()

st.title("🚨 Incident Severity Prediction: POC demo")
st.caption(f"Model in production: **{predictor.meta['model_label']}** · v{predictor.meta['version']} · trained {predictor.meta['trained_at'][:10]}")

tab1, tab2, tab3, tab4 = st.tabs(["1. Algorithm benchmark", "2. Live triage", "3. Replay unseen tickets", "4. Feedback loop"])

# ------------------------------------------------------------------ tab 1
with tab1:
    rpt = os.path.join(cfg["paths"]["reports_dir"], "benchmark_results.csv")
    if not os.path.exists(rpt):
        st.warning("Run `isp benchmark` to create the comparison.")
    else:
        res = pd.read_csv(rpt)
        sm = cfg["success_metrics"]
        st.subheader("Which algorithm is best? Scored on tickets the models never saw")
        show = res[["label", "final_macro_f1", "final_critical_miss", "final_critical_recall", "latency_ms_p95",
                    "train_seconds", "meets_f1", "meets_critical_miss", "meets_latency", "selected"]].rename(columns={
            "label": "Algorithm", "final_macro_f1": "Macro F1", "final_critical_miss": "P1 missed as P3/P4",
            "final_critical_recall": "P1 recall", "latency_ms_p95": "p95 latency (ms)", "train_seconds": "Train (s)",
            "meets_f1": f"F1 ≥ {sm['macro_f1_min']:.0%}", "meets_critical_miss": f"P1 miss ≤ {sm['critical_miss_rate_max']:.0%}",
            "meets_latency": f"< {sm['triage_seconds_max']} s", "selected": "Selected"})
        st.dataframe(show.style.format({"Macro F1": "{:.3f}", "P1 missed as P3/P4": "{:.1%}", "P1 recall": "{:.1%}",
                                        "p95 latency (ms)": "{:.0f}", "Train (s)": "{:.0f}"}), hide_index=True)
        c1, c2 = st.columns([2, 1])
        img = os.path.join(cfg["paths"]["reports_dir"], "benchmark_comparison.png")
        if os.path.exists(img):
            c1.image(img)
        img = os.path.join(cfg["paths"]["reports_dir"], "threshold_tradeoff.png")
        if os.path.exists(img):
            c2.image(img, caption="Lowering the P1 threshold catches more P1s at the cost of more false alarms")
        st.info("Selection rule: the model must keep P1 misses under the target on **validation** data, then highest macro F1 wins. "
                "Test results are only reported, never used to choose.")

# ------------------------------------------------------------------ tab 2
with tab2:
    st.subheader("Paste a ticket, get a severity in milliseconds")
    pick = st.selectbox("Load an example", ["(custom)"] + list(EXAMPLES))
    ex = EXAMPLES.get(pick, {})
    c1, c2 = st.columns(2)
    short = c1.text_input("Short description", ex.get("short_description", ""))
    desc = c1.text_area("Description", ex.get("description", ""), height=110)
    env = c2.selectbox("Environment", ["prod", "staging", "dev"], index=["prod", "staging", "dev"].index(ex.get("environment", "prod")))
    services = sorted(predictor.cmdb["affected_service"].tolist())
    svc = c2.selectbox("Affected service", services, index=services.index(ex.get("affected_service", services[0])))
    channel = c2.selectbox("Channel", ["monitoring", "portal", "email", "phone"], index=["monitoring", "portal", "email", "phone"].index(ex.get("channel", "portal")))
    users = c2.number_input("Users affected (0 = unknown)", min_value=0, value=int(ex.get("users_affected", 0)))

    if st.button("Predict severity", type="primary", disabled=not short.strip()):
        tid = f"DEMO-{int(time.time())}"
        t = dict(ticket_id=tid, short_description=short, description=desc, environment=env, affected_service=svc,
                 channel=channel, users_affected=users or np.nan)
        res = predictor.predict([t])[0]
        log_prediction(cfg["paths"]["feedback_db"], res, short)
        st.session_state["last"] = (t, res)

    if "last" in st.session_state:
        t, res = st.session_state["last"]
        a, b, c = st.columns(3)
        a.metric("Predicted severity", BADGE[res["severity"]])
        b.metric("Confidence", f"{res['confidence']:.0%}")
        c.metric("Latency", f"{res['latency_ms']:.0f} ms", help="Target is under 10 seconds")
        st.markdown(f"**Routing:** `{res['route']}`  ·  **Action:** {res['recommended_action']}")
        if res["rules_fired"]:
            st.warning(f"Business rule applied: {', '.join(res['rules_fired'])}")
        if res["escalated_by_threshold"]:
            st.warning(f"Safety threshold escalated this ticket (model's top guess was {res['model_argmax']}).")
        st.bar_chart(pd.Series(res["probabilities"], name="probability"))

        st.markdown("**Human review (feeds the learning loop)**")
        f1, f2, f3 = st.columns([1, 2, 1])
        final = f1.selectbox("Final severity", predictor.classes, index=predictor.classes.index(res["severity"]))
        reason = f2.text_input("Reason if changed", "")
        if f3.button("Submit feedback"):
            log_feedback(cfg["paths"]["feedback_db"], t["ticket_id"], res["severity"], final, "demo-user", reason)
            st.success("Saved. This becomes training data for the next model version.")

# ------------------------------------------------------------------ tab 3
with tab3:
    st.subheader("Replay recent tickets the model has never seen")
    tpath = os.path.join(cfg["paths"]["processed_dir"], "test_tickets.csv")
    if not os.path.exists(tpath):
        st.warning("Run `isp benchmark` first.")
    else:
        test = pd.read_csv(tpath, parse_dates=["created_at"])
        n = st.slider("Number of tickets", 50, min(1000, len(test)), 300, step=50)
        if st.button("Run replay"):
            sample = test.sample(n, random_state=7).reset_index(drop=True)
            t0 = time.perf_counter()
            preds = predictor.predict(sample.to_dict("records"))
            secs = time.perf_counter() - t0
            sample["predicted"] = [p["severity"] for p in preds]
            sample["route"] = [p["route"] for p in preds]
            sample["confidence"] = [p["confidence"] for p in preds]
            acc = (sample["predicted"] == sample["severity"]).mean()
            crit = sample[sample["severity"] == "P1"]
            miss = crit["predicted"].isin(["P3", "P4"]).mean() if len(crit) else float("nan")
            auto = (sample["route"] == "auto_assign").mean()
            minutes_saved = n * auto * cfg["routing"]["manual_triage_minutes"]
            k = st.columns(5)
            k[0].metric("Accuracy", f"{acc:.1%}")
            k[1].metric("P1 missed as P3/P4", f"{miss:.1%}", help=f"{len(crit)} real P1 tickets in this sample")
            k[2].metric("Auto-assigned", f"{auto:.0%}")
            k[3].metric("Avg time / ticket", f"{secs / n * 1000:.0f} ms")
            k[4].metric("Analyst time saved", f"{minutes_saved / 60:.1f} h",
                        help=f"Assumes {cfg['routing']['manual_triage_minutes']} min manual triage per ticket (set in config.yaml)")
            order = predictor.classes
            cm = pd.crosstab(sample["severity"], sample["predicted"]).reindex(index=order, columns=order, fill_value=0)
            cm.index.name, cm.columns.name = "Actual ↓", "Predicted →"
            st.markdown("**Confusion matrix**"); st.dataframe(cm)
            bad = sample[(sample["severity"] == "P1") & (sample["predicted"] != "P1")]
            st.markdown(f"**Real P1 tickets not predicted as P1 ({len(bad)})**")
            st.dataframe(bad[["ticket_id", "short_description", "environment", "predicted", "confidence", "route"]], hide_index=True)

# ------------------------------------------------------------------ tab 4
with tab4:
    st.subheader("How the model keeps learning")
    s = feedback_summary(cfg["paths"]["feedback_db"], predictor.classes)
    if s["n_feedback"] == 0:
        st.info("No feedback yet. Submit a few corrections from the Live triage tab and they appear here.")
    else:
        k = st.columns(4)
        k[0].metric("Feedback records", s["n_feedback"])
        k[1].metric("Override rate", f"{s['override_rate']:.0%}", help="Share of predictions humans changed")
        k[2].metric("Under-triage rate", f"{s['under_triage_rate']:.0%}")
        k[3].metric("P1 missed", f"{s['critical_missed']}/{s['critical_seen']}")
        st.dataframe(load_feedback(cfg["paths"]["feedback_db"]), hide_index=True)
    st.markdown("""
**Loop:** prediction logged → human confirms or corrects → final severity stored → retrain on new labels →
new model must beat the current one on a recent time-based test set (and keep P1 misses under target) → promote.
""")
