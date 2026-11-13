# Results guide

Current outputs use `artifacts/final_run/`, selected in `reports/baseline.json`.


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

The h1 results are supplementary and belong to the same published final run. Model interpretation and benchmark comparisons cover h1/h6/h24. Historical annotations should be reviewed as context; date membership is not evidence of causation.

Use the selected final run for report comparisons. Original training metadata
may retain execution-time paths; current input relocations are recorded in
`reports/baseline.json`.

## Per-coin model comparison

`reports/model_results/model_comparison_by_coin.csv` pools saved test years separately for each model, matched sample configuration, forecast horizon and coin. AP pools each coin's predictions across years and cannot be averaged to recover overall AP. Recall is blank for no targetable episodes; AP is blank for no positive hours. Median lead is conditional on true alarms. Coin-months count observed calendar months, including partial exposure.

Regenerate with `python -m modelling.reporting.by_coin`. The adjacent manifest records the source run IDs and hashes of saved fits, predictions, alarms and episode snapshots. No models are retrained.

The combined model comparison summary, yearly table, and per-coin table include h1, h6, and h24. h1 remains supplementary; filter `horizon_hours` to 6 and 24 for the primary analysis. Regenerate summary/yearly tables with `python -m modelling.reporting.model_results`; source hashes are recorded in `model_comparison_manifest.json`.

## Five headline metrics

Comparison tables order identifiers first, then `median_lead_hours`, `episode_recall`, `false_alarms`, `false_alarms_per_coin_month`, and `average_precision`, followed by supporting counts. Other fields remain available in adjacent `*_diagnostics.csv` files, joined by model/configuration/horizon/policy and applicable year or coin. Benchmark identifiers are model and horizon. Frozen run tables retain their original schemas. Event-type detection counts are a separate analysis: false alarms cannot be assigned an observed event type. See modelling/reporting/README.md for exporter scope.

## Alarm-rule robustness

`reports/robustness/alarm_smoothing/` compares original alarms with consecutive-
hour confirmation. `reports/robustness/or_combination/` compares each model pair
with its fixed-cutoff OR rule. Both contain yearly and pooled summaries and
source manifests. Models and original cutoffs are unchanged; calibration rates
are reported separately. These are distinct from alternative episode definitions.
Smoothing AP uses unchanged model scores. Binary OR AP is undefined and blank.

Final run locations and exact report-organisation wording are documented in
`docs/code_organisation.md`. Temporary validation workspaces are not part of the published outputs.

## Caught-episode cases

`reports/model_results/caught_episodes.csv` identifies each caught episode by
model, tier and horizon, with event ID, structural type, historical context and
first-warning lead time. `detection_by_historical_context.csv` compares caught
and targetable episode counts within each descriptive context. Episodes can
appear for several models or horizons.
Context annotations do not establish causation or independent statistical tests.
