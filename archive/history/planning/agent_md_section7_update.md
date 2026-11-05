# Local draft — replacement for agent.md §7 "Findings so far" (NOT pushed)

> Written 2026-10-27 from local fixed-config runs. Numbers freeze Thu 30 Oct.
> Swap this in only when the group confirms; Table B below fills from
> `reports/local_budget_curve.csv` once the sweep finishes.

## 7. Findings so far (fixed-config models, prof's guidance 2026-10-26)

**Model policy:** very little tuning. Fixed settings every year, reported on the
slides — XGBoost `max_depth=3, learning_rate=0.05, n_estimators=300`;
Random Forest `n_estimators=500, max_depth=8, min_samples_leaf=50`;
logistic `C=0.001, penalty=l1`. No per-fold grid search in final results.

**Evaluation protocol:** Alyssa's `modelling/rolling_threshold.py` — per test
year Y, fit on inner_train (years < Y−1), pick the lowest threshold on
inner_val (Y−1) meeting ≤1 false alarm per observed coin-month, apply to Y.
Headline = pooled OOS episode recall vs the false-alarm budget; PR-AUC is
secondary.

**Pooled OOS PR-AUC (fixed configs):**

| Model | h6 T1 | h6 T2 | h24 T1 | h24 T2 |
|---|---|---|---|---|
| Constant prior | 0.0020 | 0.0020 | 0.0063 | 0.0064 |
| Persistence | 0.0626 | 0.0490 | 0.0479 | 0.0338 |
| Logistic | 0.0200 | 0.0197 | 0.0312 | 0.0300 |
| Random Forest | 0.0593 | 0.0413 | **0.0685** | **0.0454** |
| XGBoost | 0.0522 | **0.0598** | 0.0480 | 0.0412 |

Random Forest leads at h=24; fixed XGB beats its old recall-tuned numbers
everywhere; persistence stays the best single rule at h=6.

**Episode recall at the ≤1 FA/coin-month budget (pooled OOS):**

| Model | h6 T1 | h6 T2 | h24 T1 | h24 T2 |
|---|---|---|---|---|
| Persistence | 0.15 | 0.15 | 0.26 | 0.26 |
| Logistic | 0.15 | 0.12 | 0.38 | 0.32 |
| Random Forest | 0.06 | 0.12 | 0.21 | 0.29 |
| XGBoost | 0.12 | 0.12 | 0.38 | **0.44** |

XGBoost Tier 2 leads at h=24 (15/34 episodes); RF catches up at budgets 4–8
(0.56/0.71 vs XGB 0.53/0.65). Honest warts: 2022 is a universal 0/9 for every
model (even persistence) and XGB catches only 2/9 even at 8 FA/coin-month —
Terra/LUNA had no 24h market/on-chain precursor; 2021 false alarms overshoot
the budget for all models (XGB 3.11 FA/coin-month vs 1.0 target) because the
budget is enforced at calibration, not in test. Full sweep: `reports/local_budget_curve.csv`,
per-year: `reports/local_budget_per_year.csv`.

## Gotcha 11 (new)

- Fixed-config runs are tagged `rf_500_d8_l50`, `logreg_c0001_l1`,
  `pers_fixed`, `base_fixed`, `xgb_d3_lr005_300`; `train.py --model-params`
  pins the settings so nobody needs to edit a model class to follow the
  prof's no-tuning rule.
