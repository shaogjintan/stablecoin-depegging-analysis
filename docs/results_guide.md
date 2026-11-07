# Results guide

| Report claim or exhibit | Authoritative output |
|---|---|
| Model performance, matched tiers, h1/h6/h24 | reports/model_results/model_comparison_summary.csv |
| Test-year performance | reports/model_results/model_comparison_by_year.csv |
| Validation exposure, cutoff and false alarms | reports/validation_checks/calibration_audit.csv |
| Constant-prior and persistence comparison | reports/model_results/benchmark.csv |
| Alternative episode definitions | reports/robustness/alt_definitions/metrics.csv |
| Tier tradeoffs under alternatives | reports/robustness/alt_definitions/onchain_effects.csv |
| Grouped events and historical context | reports/depeg_events/episode_event_mapping.csv |
| Structural event types | reports/depeg_events/event_classification.csv |
| Final detections by type (counts) | reports/model_results/model_comparison_by_event_type.csv |
| Source completeness with explicit denominator | reports/data_coverage/coverage.csv |
| Final LR coefficients | reports/model_interpretation/log_reg/lr_coefficients.csv |
| Final RF importance | reports/model_interpretation/random_forest/rf_importance_by_year.csv |
| Final XGB SHAP ranking/figures | reports/model_interpretation/ |

The h1 results are supplementary, retained under artifacts/runs/supplementary/. Model interpretation figures and benchmark comparisons cover primary h6/h24. Counts by type also cover h1. Historical annotations must be reviewed as context; date membership is not evidence of causation.

All old final_results/, fixed/tuned figures and non-refit coefficients are archived. Do not combine their values with the current final-model outputs. Frozen JSON manifests may contain old paths; resolve them with docs/cleanup_moves.json rather than editing the original provenance.

## Per-coin model comparison

`reports/model_results/model_comparison_by_coin.csv` pools saved test years separately for each model, matched sample configuration, forecast horizon and coin. It follows the scope of the current overall summary. Counts sum back to that summary; recall and false-alarm rates use each coin's own denominators. AP pools each coin's predictions across years and cannot be averaged to recover overall AP. Recall is blank for no targetable episodes; AP is blank for no positive hours. Median lead is conditional on true alarms. Coin-months count observed calendar months, including partial exposure.

Regenerate with `python -m modelling.reporting.by_coin`. The adjacent manifest records the source run IDs and hashes of saved fits, predictions, alarms and episode snapshots. No models are retrained.

The combined model comparison summary, yearly table, and per-coin table include h1, h6, and h24. h1 remains supplementary; filter `horizon_hours` to 6 and 24 for the primary analysis. Regenerate summary/yearly tables with `python -m modelling.reporting.model_results`; source hashes are recorded in `model_comparison_manifest.json`.

## Five headline metrics

Comparison tables order identifiers first, then `median_lead_hours`, `episode_recall`, `false_alarms`, `false_alarms_per_coin_month`, and `average_precision`, followed by supporting counts. Other fields remain available in adjacent `*_diagnostics.csv` files, joined by model/configuration/horizon/policy and applicable year or coin. Benchmark identifiers are model and horizon. Frozen run tables retain their original schemas. Event-type detection counts are a separate analysis: false alarms cannot be assigned an observed event type. See modelling/reporting/README.md for exporter scope.
