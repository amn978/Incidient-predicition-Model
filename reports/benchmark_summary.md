# Benchmark results (test set = newest tickets)

Targets: macro F1 >= 80%, P1 missed as P3/P4 <= 2%, latency <= 10 s.

| Model | Macro F1 | P1 miss % | P1 recall | p95 ms | Train s | Meets all |
|---|---|---|---|---|---|---|
| Ensemble (LogReg + LightGBM) (selected) | 0.787 | 0.0% | 81.8% | 68 | 27 | no |
| Logistic Regression (TF-IDF) | 0.775 | 0.0% | 87.8% | 26 | 13 | no |
| LightGBM | 0.766 | 0.0% | 78.4% | 35 | 15 | no |
| Linear SVM (TF-IDF, calibrated) | 0.752 | 0.0% | 81.1% | 39 | 10 | no |
| XGBoost | 0.746 | 0.0% | 77.7% | 35 | 60 | no |
| Random Forest | 0.695 | 0.0% | 79.1% | 94 | 4 | no |
| Complement Naive Bayes | 0.584 | 1.4% | 69.6% | 26 | 3 | no |
