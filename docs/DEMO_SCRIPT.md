# 10-minute demo script for your manager

Start the UI: `streamlit run app/streamlit_app.py`

## 0:00 Problem (1 min), say it, no clicks
"Severity is set by hand: slow, inconsistent, and a missed P1 delays response and breaks SLAs.
This POC predicts severity at ticket creation, in milliseconds, and sends doubtful cases to a person."

## 1:00 Tab 1, Algorithm benchmark (3 min)
- "We trained 7 algorithms on the same data and tested on the **newest** tickets they never saw."
- Point at the green bar: the model is **selected on validation data only**, so the test result is honest.
- Walk the three targets: **P1 misses** (target < 2%), **latency** (target < 10 s), **macro F1** (target 80%).
- Show the trade-off chart: "Lowering the P1 threshold catches more P1s but raises false alarms. We tune this
  deliberately, because a missed P1 costs far more than a false alarm."
- Be upfront about any target not met. Use `reports/data_quality_sensitivity.csv`: the limit is how much
  context the ticket captures, not the algorithm.

## 4:00 Tab 2, Live triage (3 min)
1. Load **"Prod primary DB down"** → P1, rule banner, "page on-call" action, latency in ms.
2. Load **"Leaked access keys in prod"** → P1 via business rule (rules live in config, no retraining).
3. Load **"5xx spike on customer portal"** → P2.
4. Load **"Pipeline failure in dev"** → P4, auto-assigned.
5. Edit the description (e.g., change environment to dev) and show severity change.
6. Submit feedback with a different severity: "this becomes training data."

## 7:00 Tab 3, Replay unseen tickets (2 min)
- Run 300 tickets: accuracy, **P1 missed**, auto-assign rate, time per ticket, analyst hours saved.
- Say clearly that hours saved uses an **assumed 5 min/ticket** (change `manual_triage_minutes` in config).
- Show the list of any real P1s the model did not call P1, to demonstrate error analysis.

## 9:00 Tab 4, Feedback loop and next steps (1 min)
- Override rate, under-triage rate and P1-missed counts are the monitoring signals that trigger retraining.
- Next: connect to ServiceNow/Jira, run on real tickets, shadow-mode for 2 to 4 weeks, then limited auto-assign.

## Likely questions
- **"Is this real data?"** No, synthetic, to show the method. Real results come from a one-week run on your exports.
- **"Why not a big LLM?"** Cost, latency and consistency. Fast classical models meet the latency target and
  give calibrated probabilities, which the P1 safety thresholds need. LLMs remain an option as a second opinion.
- **"What if it is wrong on a P1?"** Safety thresholds + rules + human review band, and every miss is logged and learned from.
