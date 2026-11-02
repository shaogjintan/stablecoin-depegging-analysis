# Consolidated presentation values — refit after calibration

Updated 2026-11-02T11:26:51.735502+00:00. Source run: `runs/REFIT-LOWEST-NEW` (run ID `f2c840d7-2ff9-4552-84fe-cbc4c616ea5d`).

Corrected_v1; matched Tier 1/2 eligible samples; test years 2021–2025; h6/h24. Practice-model calibration followed by a full-training-window refit. These values replace the earlier no-refit values.

Fixed RF: 500 trees, max depth 8, minimum leaf size 50. Fixed XGBoost: 300 trees, max depth 3, learning rate 0.05; no early stopping. Other settings are recorded in each fit.json.

LogReg source: `runs/LR-REFIT-LOWEST-NEW`; status: **complete**. Partial runs are excluded.

## Slide 1 — Random forest impurity importance (Tier 2)

| Feature/group | h6 | h24 |
| --- | --- | --- |
| realised_vol_24h | 22.5% | 18.3% |
| peg_stress | 8.4% | 12.0% |
| peg_deviation | 8.5% | 12.1% |
| realised_vol_6h | 16.4% | 14.7% |
| other_market | 21.6% | 22.9% |
| onchain | 22.7% | 20.2% |

Mean normalized impurity importance across five final refitted forests. On-chain features are 12/22 = 54.5% of inputs. Importance is training attribution; it is not the percentage improvement in forecasting.

## Slide 2 — XGBoost SHAP (Tier 2)

h24 on-chain: **26.01%**; market: **73.99%**. h6 on-chain: **24.68%**.

| h24 feature | Share of total absolute SHAP |
| --- | --- |
| realised_vol_24h | 19.76% |
| realised_vol_6h | 13.68% |
| peg_deviation | 12.00% |
| tx_volume_logdiff_24h | 3.42% |
| active_receivers_logdiff_24h | 3.31% |
| tx_volume_logdiff_1h | 3.11% |

Top three market and top three on-chain features shown. Exact TreeSHAP raw log-odds attribution over every eligible out-of-sample row, using the final refitted models. Attribution does not establish forecasting gains or causality. Regenerate donut geometry with the new shares.

## Slide 3 — Threshold methodology

For a 2024 test example:

1. Fit the practice model on eligible 2018–2022 rows.
2. Score held-out 2023.
3. Evaluate every distinct validation score plus the no-alarm cutoff; choose the lowest cutoff with grouped false alarms / observed coin-months ≤ 1.
4. Refit a fresh model on full eligible 2018–2023 training rows and apply the unchanged cutoff to 2024. Preserve purge/embargo and episode-boundary exclusions.

The budget is 72 only if six coins each have 12 observed months; a month with any eligible rows counts once, without hourly prorating. The budget is pooled, not a separate hard limit for each coin-month. Refit can change score scales; validation feasibility does not guarantee the realised test rate. For LogReg, C is selected by calendar-year CV inside practice training, then held fixed for refitting.

| Test year | Calibration year | Observed calibration coin-months | False-alarm budget |
| --- | --- | --- | --- |
| 2021 | 2020 | 72 | 72 |
| 2022 | 2021 | 72 | 72 |
| 2023 | 2022 | 63 | 63 |
| 2024 | 2023 | 66 | 66 |
| 2025 | 2024 | 59 | 59 |

## Slide 4 — Tier 1 performance (matched sample)

| h | Model | Pooled AP | Positive base rate (%) | False alarms | FA / coin-month | Episode recall | Median lead (hours) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 6 | logreg | 0.006823 | 0.1019 | 387 | 1.225 | 9/33 (27.27%) | 4.0 |
| 6 | random_forest | 0.005201 | 0.1019 | 230 | 0.728 | 4/33 (12.12%) | 3.5 |
| 6 | xgboost | 0.005264 | 0.1019 | 303 | 0.959 | 6/33 (18.18%) | 2.5 |
| 24 | logreg | 0.018550 | 0.4071 | 331 | 1.047 | 12/34 (35.29%) | 6.5 |
| 24 | random_forest | 0.017959 | 0.4071 | 267 | 0.845 | 8/34 (23.53%) | 8.0 |
| 24 | xgboost | 0.014811 | 0.4071 | 296 | 0.937 | 6/34 (17.65%) | 5.5 |

## Slide 5 — Tier 1 findings

h6: most episodes caught by logreg (9/33). h24: most episodes caught by logreg (12/34).

Lead times are medians conditional on caught episodes; different models can catch different episodes.

## Slide 6 — Tier 2 performance

| h | Model | Pooled AP | Positive base rate (%) | False alarms | FA / coin-month | Episode recall | Median lead (hours) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 6 | logreg | 0.006864 | 0.1019 | 398 | 1.259 | 7/33 (21.21%) | 4.0 |
| 6 | random_forest | 0.004868 | 0.1019 | 217 | 0.687 | 2/33 (6.06%) | 2.5 |
| 6 | xgboost | 0.004387 | 0.1019 | 421 | 1.332 | 5/33 (15.15%) | 3.0 |
| 24 | logreg | 0.018336 | 0.4071 | 316 | 1.000 | 12/34 (35.29%) | 6.0 |
| 24 | random_forest | 0.017289 | 0.4071 | 184 | 0.582 | 4/34 (11.76%) | 12.5 |
| 24 | xgboost | 0.016315 | 0.4071 | 245 | 0.775 | 6/34 (17.65%) | 9.0 |

## Slide 7 — Tier 2 and on-chain findings

h6: most episodes caught by logreg (7/33). h24: most episodes caught by logreg (12/34).

| Model | h | Episodes caught T1 → T2 | False alarms T1 → T2 | AP T1 → T2 |
| --- | --- | --- | --- | --- |
| random_forest | 6 | 4 → 2 | 230 → 217 | 0.005201 → 0.004868 |
| random_forest | 24 | 8 → 4 | 267 → 184 | 0.017959 → 0.017289 |
| xgboost | 6 | 6 → 5 | 303 → 421 | 0.005264 → 0.004387 |
| xgboost | 24 | 6 → 6 | 296 → 245 | 0.014811 → 0.016315 |
| logreg | 6 | 9 → 7 | 387 → 398 | 0.006823 → 0.006864 |
| logreg | 24 | 12 → 12 | 331 → 316 | 0.018550 → 0.018336 |

Yearly context: random_forest h24 in 2021: Tier 1 catches 5 versus Tier 2's 0, with 158 versus 66 false alarms; xgboost h6 in 2021: Tier 1 catches 3 versus Tier 2's 1, with 158 versus 264 false alarms. The earliest test fold has the shortest training history. This describes where the differences arise, without proving why. Full yearly counts: model_comparison_by_year.csv.

Compare the recall/false-alarm trade-off and AP together. Both tiers use matching training, calibration and test rows, with their own calibrated thresholds. These are observed sample differences, without a statistical significance claim.

## Slide 8 — Episodes and events caught

Episode timestamps are UTC bar closes. Event IDs are chronological groups with a 48-hour anchored window and no repeated coin; they describe co-occurrence rather than establish a historical cause.

### Tier 2

| h | Model | Episodes caught | Events caught | By coin | By year |
| --- | --- | --- | --- | --- | --- |
| 6 | logreg | PAX 2021-01-05 01:00 UTC; TUSD 2021-01-05 01:00 UTC; BUSD 2021-04-19 16:00 UTC; TUSD 2021-05-19 12:00 UTC; USDT 2021-08-06 02:00 UTC; DAI 2023-03-11 02:00 UTC; TUSD 2024-01-15 21:00 UTC | event 27 (2021-01-05); event 29 (2021-04-19); event 32 (2021-05-17); event 36 (2021-08-06); event 51 (2023-03-11); event 54 (2024-01-15) | BUSD: 1; DAI: 1; PAX: 1; TUSD: 3; USDT: 1 | 2021: 5; 2023: 1; 2024: 1 |
| 6 | random_forest | BUSD 2021-04-19 16:00 UTC; TUSD 2024-01-15 21:00 UTC | event 29 (2021-04-19); event 54 (2024-01-15) | BUSD: 1; TUSD: 1 | 2021: 1; 2024: 1 |
| 6 | xgboost | BUSD 2021-04-19 16:00 UTC; BUSD 2022-01-21 22:00 UTC; USDC 2022-01-21 22:00 UTC; TUSD 2024-01-15 21:00 UTC; TUSD 2024-11-14 15:00 UTC | event 29 (2021-04-19); event 40 (2022-01-21); event 54 (2024-01-15); event 55 (2024-11-14) | BUSD: 2; TUSD: 2; USDC: 1 | 2021: 1; 2022: 2; 2024: 2 |
| 24 | logreg | DAI 2021-01-05 01:00 UTC; PAX 2021-01-05 01:00 UTC; TUSD 2021-01-05 01:00 UTC; TUSD 2021-01-11 04:00 UTC; BUSD 2021-04-19 16:00 UTC; TUSD 2021-05-19 12:00 UTC; TUSD 2021-05-21 15:00 UTC; USDT 2021-08-06 02:00 UTC; DAI 2023-03-11 02:00 UTC; TUSD 2024-01-15 21:00 UTC; TUSD 2024-11-14 15:00 UTC; TUSD 2025-06-12 23:00 UTC | event 27 (2021-01-05); event 28 (2021-01-11); event 29 (2021-04-19); event 32 (2021-05-17); event 33 (2021-05-21); event 36 (2021-08-06); event 51 (2023-03-11); event 54 (2024-01-15); event 55 (2024-11-14); event 56 (2025-06-12) | BUSD: 1; DAI: 2; PAX: 1; TUSD: 7; USDT: 1 | 2021: 8; 2023: 1; 2024: 2; 2025: 1 |
| 24 | random_forest | BUSD 2022-01-21 22:00 UTC; TUSD 2024-01-15 21:00 UTC; TUSD 2024-11-14 15:00 UTC; TUSD 2025-06-12 23:00 UTC | event 40 (2022-01-21); event 54 (2024-01-15); event 55 (2024-11-14); event 56 (2025-06-12) | BUSD: 1; TUSD: 3 | 2022: 1; 2024: 2; 2025: 1 |
| 24 | xgboost | TUSD 2021-01-11 04:00 UTC; TUSD 2021-05-19 12:00 UTC; BUSD 2022-01-21 22:00 UTC; TUSD 2024-01-15 21:00 UTC; TUSD 2024-11-14 15:00 UTC; TUSD 2025-06-12 23:00 UTC | event 28 (2021-01-11); event 32 (2021-05-17); event 40 (2022-01-21); event 54 (2024-01-15); event 55 (2024-11-14); event 56 (2025-06-12) | BUSD: 1; TUSD: 5 | 2021: 2; 2022: 1; 2024: 2; 2025: 1 |

### Tier 1 matched

| h | Model | Episodes caught | Events caught | By coin | By year |
| --- | --- | --- | --- | --- | --- |
| 6 | logreg | PAX 2021-01-05 01:00 UTC; TUSD 2021-01-05 01:00 UTC; BUSD 2021-04-19 16:00 UTC; TUSD 2021-05-19 12:00 UTC; USDT 2021-08-06 02:00 UTC; DAI 2023-03-11 02:00 UTC; TUSD 2024-01-15 21:00 UTC; TUSD 2024-11-14 15:00 UTC; TUSD 2025-06-12 23:00 UTC | event 27 (2021-01-05); event 29 (2021-04-19); event 32 (2021-05-17); event 36 (2021-08-06); event 51 (2023-03-11); event 54 (2024-01-15); event 55 (2024-11-14); event 56 (2025-06-12) | BUSD: 1; DAI: 1; PAX: 1; TUSD: 5; USDT: 1 | 2021: 5; 2023: 1; 2024: 2; 2025: 1 |
| 6 | random_forest | BUSD 2021-04-19 16:00 UTC; BUSD 2022-01-21 22:00 UTC; TUSD 2024-01-15 21:00 UTC; TUSD 2024-11-14 15:00 UTC | event 29 (2021-04-19); event 40 (2022-01-21); event 54 (2024-01-15); event 55 (2024-11-14) | BUSD: 2; TUSD: 2 | 2021: 1; 2022: 1; 2024: 2 |
| 6 | xgboost | BUSD 2021-04-19 16:00 UTC; USDC 2021-04-25 08:00 UTC; USDT 2021-08-06 02:00 UTC; BUSD 2022-01-21 22:00 UTC; TUSD 2024-01-15 21:00 UTC; TUSD 2024-11-14 15:00 UTC | event 29 (2021-04-19); event 30 (2021-04-25); event 36 (2021-08-06); event 40 (2022-01-21); event 54 (2024-01-15); event 55 (2024-11-14) | BUSD: 2; TUSD: 2; USDC: 1; USDT: 1 | 2021: 3; 2022: 1; 2024: 2 |
| 24 | logreg | DAI 2021-01-05 01:00 UTC; PAX 2021-01-05 01:00 UTC; TUSD 2021-01-05 01:00 UTC; USDC 2021-01-05 01:00 UTC; TUSD 2021-01-11 04:00 UTC; TUSD 2021-05-19 12:00 UTC; TUSD 2021-05-21 15:00 UTC; USDT 2021-08-06 02:00 UTC; DAI 2023-03-11 02:00 UTC; TUSD 2024-01-15 21:00 UTC; TUSD 2024-11-14 15:00 UTC; TUSD 2025-06-12 23:00 UTC | event 27 (2021-01-05); event 28 (2021-01-11); event 32 (2021-05-17); event 33 (2021-05-21); event 36 (2021-08-06); event 51 (2023-03-11); event 54 (2024-01-15); event 55 (2024-11-14); event 56 (2025-06-12) | DAI: 2; PAX: 1; TUSD: 7; USDC: 1; USDT: 1 | 2021: 8; 2023: 1; 2024: 2; 2025: 1 |
| 24 | random_forest | TUSD 2021-01-05 01:00 UTC; USDC 2021-01-05 01:00 UTC; USDT 2021-01-05 01:00 UTC; TUSD 2021-01-11 04:00 UTC; BUSD 2021-04-19 16:00 UTC; TUSD 2024-01-15 21:00 UTC; TUSD 2024-11-14 15:00 UTC; TUSD 2025-06-12 23:00 UTC | event 27 (2021-01-05); event 28 (2021-01-11); event 29 (2021-04-19); event 54 (2024-01-15); event 55 (2024-11-14); event 56 (2025-06-12) | BUSD: 1; TUSD: 5; USDC: 1; USDT: 1 | 2021: 5; 2024: 2; 2025: 1 |
| 24 | xgboost | USDT 2021-01-05 01:00 UTC; TUSD 2021-01-11 04:00 UTC; BUSD 2021-04-19 16:00 UTC; TUSD 2024-01-15 21:00 UTC; TUSD 2024-11-14 15:00 UTC; TUSD 2025-06-12 23:00 UTC | event 27 (2021-01-05); event 28 (2021-01-11); event 29 (2021-04-19); event 54 (2024-01-15); event 55 (2024-11-14); event 56 (2025-06-12) | BUSD: 1; TUSD: 4; USDT: 1 | 2021: 3; 2024: 2; 2025: 1 |

## Slide 9 — Benchmarks on the same common sample

| h | Model | Pooled AP | Positive base rate (%) | False alarms | FA / coin-month | Episode recall | Median lead (hours) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 6 | baseline | 0.001252 | 0.1019 | 0 | 0.000 | 0/33 (0.00%) | — |
| 6 | persistence | 0.010259 | 0.1019 | 289 | 0.915 | 5/33 (15.15%) | 4.0 |
| 24 | baseline | 0.004958 | 0.4071 | 0 | 0.000 | 0/34 (0.00%) | — |
| 24 | persistence | 0.014930 | 0.4071 | 273 | 0.864 | 9/34 (26.47%) | 10.0 |

Benchmarks follow the same practice-calibration/full-refit protocol. Their values are identical across the matched tiers. Constant prior varies by training year, so pooled AP need not equal pooled prevalence. Compare models at their realised false-alarm rates; do not assert uniform superiority from recall alone.

## Shared footnote

Pooled OOS 2021–2025; corrected_v1; identical eligible coin-hours across tiers; AP = sklearn.average_precision_score (not trapezoidal PR-AUC); exhaustive lowest-feasible practice-model cutoff at ≤1 false alarm / observed coin-month, transferred unchanged to a full-window refit. Median lead conditional on caught episodes.

## Refresh after LogReg completes

`python -m modelling.consolidate_refit_slides --run-dir runs/REFIT-LOWEST-NEW --lr-run runs/LR-REFIT-LOWEST-NEW --reuse-shap`

h1 is deliberately excluded from this slide update and is run afterward in a separate output directory.

Completed h1 RF/XGBoost results (run after the h6/h24 slide update): `runs/REFIT-LOWEST-H1/summary.csv`.
