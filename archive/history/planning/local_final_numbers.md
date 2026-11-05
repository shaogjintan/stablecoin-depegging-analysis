# LOCAL-ONLY Final Numbers for Deck P18/P19 (fixed-config, prof's settings)

> Protocol: Alyssa's `rolling_threshold.py` — model fit on inner_train (years < Y−1),
> threshold chosen on inner_val (year Y−1) at ≤1 FA/coin-month, applied to test year Y.
> Model configs are FIXED (prof's guidance): XGB `d3_lr005_300`, RF `500 trees/depth 8/leaf 50`,
> logreg `C=0.001 L1`. No tuning grids. Nothing here is pushed.

## Fixed hyperparameters (P14 — report verbatim)

| Model | Settings |
|---|---|
| XGBoost | `max_depth=3, learning_rate=0.05, n_estimators=300, min_child_weight=1, scale_pos_weight="auto", tree_method="hist", random_state=42` |
| Random Forest | `n_estimators=500, max_depth=8, min_samples_leaf=50, max_features="sqrt", class_weight="balanced_subsample", random_state=42` |
| Logistic | `C=0.001, penalty="l1", class_weight="balanced", winsor=0.01` |
| Persistence / baseline | no parameters |

## P18 Table A — PR-AUC (pooled OOS, fixed-config runs)

| Model | h6 T1 | h6 T2 | h24 T1 | h24 T2 |
|---|---|---|---|---|
| Constant prior | 0.0020 | 0.0020 | 0.0063 | 0.0064 |
| Persistence | **0.0626** | 0.0490 | 0.0479 | 0.0338 |
| Logistic (C=0.001 L1) | 0.0200 | 0.0197 | 0.0312 | 0.0300 |
| Random Forest (500/d8/l50) | 0.0593 | 0.0413 | **0.0685** | **0.0454** |
| XGBoost (d3/lr.05/300) | 0.0522 | **0.0598** | 0.0480 | 0.0412 |

**Headline:** with the prof's fixed settings, **Random Forest leads PR-AUC at h=24** (0.0685 T1 /
0.0454 T2) and XGBoost's fixed config beats its old tuned numbers everywhere (t1 h24: 0.048 vs 0.0215).
Persistence remains the best single rule at h=6. NOTE: Javier's P16 logreg numbers (0.0241/0.0220/
0.0356/0.0307) came from his grid run — replace with the fixed-config rows above for consistency.

_Source: `results/evaluation/` outputs of the local fixed-config train.py runs._

## P18 Table B — Episode recall @ ≤1 FA/coin-month budget (pooled OOS, 33/34 episodes)

| Model | h6 T1 | h6 T2 | h24 T1 | h24 T2 |
|---|---|---|---|---|
| Constant prior | 0.00 | 0.00 | 0.00 | 0.00 |
| Persistence | 0.15 | 0.15 | 0.26 | 0.26 |
| Logistic | 0.15 | 0.12 | 0.38 | 0.32 |
| Random Forest | 0.06 | 0.12 | 0.21 | 0.29 |
| XGBoost | 0.12 | 0.12 | **0.38** | **0.44** |

**Headline:** at the operating budget, **XGBoost Tier 2 leads at h=24 (15/34 = 0.44)**;
logreg ties it on Tier 1 (13/34). RF — the PR-AUC leader — catches up on the curve at
budgets 4–8 (h24 T2: RF 0.56 @4, 0.71 @8 vs XGB 0.53/0.65). Persistence stays respectable
(0.74 @8). So: **RF ranks best, XGBoost alarms best at the budget.**

_Source: `reports/local_budget_curve.csv` (budget 1.0 rows), reproduced against the committed XGB rolling CSVs._

## P18 Figure — Recall vs false-alarm budget curve (h=24)

`reports/local_budget_curve_t1_h24.png` and `..._t2_h24.png` (log-x, budgets 0.5–8 FA/coin-month, all 5 models).

## P18 Honest warts (acknowledge on the slide — now evidence-backed)

1. **2022 is a universal miss — but it is the JANUARY 2022 wobble, not Terra.**
   Our 2022 episodes are all Jan 5 – Feb 11, 2022 (market-wide wobble; 9 targetable) plus
   Nov 13, 2022 (PAX/TUSD, not targetable). At h=24 T2 every model catches **0/9** of those
   at the 1-FA budget; even at **8 FA/coin-month** XGBoost catches only 2/9.
   The May 2022 Terra-era dips appear in this price series only as **single-hour wicks**
   (BUSD $0.9695 for 1h, USDT $0.9876 for 1h) and are excluded by the ≥2-consecutive-hour
   episode rule — so Terra is outside our scored set, not a model failure.
   → Slide line: "2022: 0/9 — every model missed the January 2022 market-wide wobble."
   → P21 limitation: single-hour Terra-era wicks fall outside the 2-hour episode definition.
2. **2021 false alarms overshoot the budget for every model** (XGB 3.11, logreg 2.51, RF 2.01,
   persistence 2.04 FA/coin-month vs the 1.0 target). The budget is enforced at calibration
   (Y−1) but drifts in volatile test years. → P21 limitation; rank-based thresholds = future work.
3. Fixed-config XGB h24 T2 thresholds are high (0.59–0.91) — absolute probabilities don't
   transfer across years; that's the mechanism behind both warts.

## P19 — Tier 1 vs Tier 2 (SHAP story, fixed-config XGB)

- **On-chain share of |SHAP|: 20.7% (h=24), 17.7% (h=6)** — the prof's fixed config finds
  MORE on-chain signal than the old recall-tuned runs (13.2% / 11.1%). Good for the story.
- Top features h=24 T2: `realised_vol_24h` 21.8%, `realised_vol_6h` 13.3%, `peg_deviation`
  12.1%, `peg_stress` 9.2%; top on-chain: `tx_volume_logdiff_24h` 2.7%,
  `active_senders_logdiff_24h` 2.7%, `active_receivers_logdiff_24h` ~2.5%.
- Figures to paste: `reports/figures/group_share_xgboost_t2_h24_fixed.png` (donut) +
  `shap_ranking_xgboost_t2_h24_fixed.png` (bar); backup `beeswarm_xgboost_t2_h24_fixed.png`.
- Recall-cost line (T2 vs T1 at h=24, budget 1.0): XGB T2 0.44 vs T1 0.38 — on-chain adds
  ~6pp recall at the same budget; at budget 8: T2 0.65 vs T1 0.68 (wash).
- Matched coin-hours rerun: **deferred** (keep Javier's † note on P16/P19).
- Regeneration command (uses a local path-fixed copy of the script):
  `python local_shap/shap_analysis.py --model xgboost --horizon 24 --tier 2 --tag xgboost_t2_h24_fixed`

## Where the numbers live

- `reports/local_budget_curve.csv` — budget sweep (all models, h6/h24, t1/t2).
- `reports/rolling_thresholds/` — committed XGB CSVs + local RF/logreg/persistence/baseline CSVs.
- `results/evaluation/` — pooled PR-AUC etc. from local train.py runs.
