# Incident Severity Prediction (POC → production-ready skeleton)

Predicts the severity (P1 Critical … P4 Low) of an IT / cloud incident **at the moment the ticket is
created**, compares several algorithms against the POC success metrics, and serves the best model through
an API and a manager-friendly demo UI.

```
ticket → ingestion API → feature pipeline → ML model → safety thresholds → business rules
       → auto-assign  OR  human triage queue → ITSM write-back → feedback store → retraining
```

## What is in the box

| Path | Purpose |
|---|---|
| `config/config.yaml` | All settings: targets, split, thresholds, routing, business rules |
| `src/incident_severity/data_generator.py` | Synthetic cloud-incident tickets + CMDB (swap for real ITSM export) |
| `src/incident_severity/features.py` | Text cleaning, PII masking, structured/time features |
| `src/incident_severity/models.py` | Model zoo: LogReg, Linear SVM, Complement NB, Random Forest, LightGBM, XGBoost, Ensemble, (optional) sentence-embeddings |
| `src/incident_severity/thresholds.py` | Safety thresholds that trade precision for P1 recall |
| `src/incident_severity/rules.py` | Config-driven keyword rules (can only raise severity) |
| `src/incident_severity/benchmark.py` | Trains everything, scores against the 3 success metrics, saves the best |
| `src/incident_severity/predict.py` / `api.py` | Inference class + FastAPI service |
| `src/incident_severity/storage.py` | Prediction log + feedback store (SQLite demo, PostgreSQL in prod) |
| `app/streamlit_app.py` | Demo UI (benchmark, live triage, replay, feedback loop) |
| `scripts/data_quality_sensitivity.py` | Shows how data quality limits achievable F1 |
| `tests/` | 12 unit and end-to-end tests (incl. API) |

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e .                                          # installs the `isp` command

isp generate-data          # 1. create 12,000 synthetic tickets + CMDB   -> data/
isp benchmark              # 2. train + compare all algorithms           -> models/, reports/  (~4 min on 1 CPU)
pytest -q                  # 3. optional: run the tests
streamlit run app/streamlit_app.py                       # 4. manager demo UI
uvicorn incident_severity.api:app --port 8000            # 5. REST API  (docs at http://localhost:8000/docs)
```

Optional semantic model: `pip install sentence-transformers` then `isp benchmark --embeddings`.

## How the design maps to the success metrics

| POC metric | How it is addressed | Where |
|---|---|---|
| Macro F1 ≥ 80% | Class-balanced training, rich text + structured + CMDB features, 7 algorithms compared, time-based test set | `models.py`, `benchmark.py` |
| P1 missed as P3/P4 < 2% | Probability output + validation-tuned safety thresholds + keyword rules + human-review band | `thresholds.py`, `rules.py`, `predict.py` |
| Triage < 10 seconds | Measured end-to-end p95 latency per ticket (features + model + rules) | `benchmark.py` |

**Metric definitions**
- *P1 missed* = real P1 tickets predicted as P3 or P4, divided by all real P1 tickets (95% upper bound reported too, because P1 counts are small).
- *Macro F1* = unweighted mean of F1 over P1..P4, so rare classes count as much as common ones.
- *Latency* = single-ticket prediction including cleaning, features, model, thresholds, rules.

## Engineering practices built in

- **No leakage:** columns known only after triage (`resolution_minutes`, `closed_at`, `assignment_group`, `close_notes`) are dropped (`config.yaml → leakage_columns`).
- **Time-based split:** oldest 60% train, next 20% validation, newest 20% test. Nothing is shuffled.
- **No test peeking:** thresholds are fitted and the best model is chosen on validation only.
- **No train/serve skew:** feature building lives inside the sklearn pipeline, so training and API use identical code.
- **PII masking:** emails, IPs, ticket ids and hex ids are masked before modelling.
- **Safety net:** low-confidence predictions go to a human review queue; rules can only raise severity.
- **Feedback loop:** every prediction is logged, and human corrections are stored for monitoring and retraining.
- **Versioned artifacts:** `models/best_model.joblib` + `models/model_card.json` (metrics, thresholds, version, timestamp).

## Using real data

1. Export tickets to `data/tickets.csv` with at least: `ticket_id, created_at, channel, category, subcategory,
   affected_service, environment (prod/staging/dev), short_description, description, users_affected, severity (P1..P4)`.
2. Export the service catalog to `data/cmdb.csv`: `affected_service, service_tier (1-3), customer_facing (0/1)`.
3. List every post-resolution column in `leakage_columns`.
4. Run `isp benchmark`. Review label quality first: if two analysts disagree on 20% of tickets, that caps F1.
5. Adapt `rules.items` to your own runbooks, and replace `manual_triage_minutes` with your measured value.

## Roadmap to production

1. Read tickets via ITSM webhook (ServiceNow/Jira) and write severity back via REST.
2. Move the feedback store to PostgreSQL; schedule retraining (Airflow/cron) with a promotion gate (new model must beat the current one on a recent time window and keep the P1 miss target).
3. Add shadow mode (run the new model silently before promoting), drift monitoring (Evidently), metrics (Prometheus/Grafana).
4. Add SHAP explanations to the review UI; add authentication (API key support already included via `ISP_API_KEY`).
5. Fine-tune a transformer only if the baseline cannot reach the F1 target on real data.

## Honest limitations

- The demo data is **synthetic**. Results show the method and the pipeline, not what your real tickets will give.
- Label noise and missing context (impact, users affected) cap achievable F1. See `reports/data_quality_sensitivity.csv`.
- A few hundred P1 examples in test means the P1 miss rate has wide uncertainty. Report the confidence bound, not only the point estimate.
