# Reporting

Run exporters as modules from the repository root. `python -m reports.reproduce` runs the saved-evidence reporting workflow; it does not refit primary models. The selected evidence is `artifacts/final_run/`, with canonical inputs under `data/`. Outputs go under `reports/`.

## Performance comparison layout

Summary, yearly, per-coin, and benchmark comparison tables use this order:

1. Identifiers: model, configuration, horizon, and applicable year/coin/policy.
2. Five headline metrics: median lead hours, episode recall, false alarms, false alarms per observed coin-month, and average precision.
3. Supporting counts: episodes caught/targetable, observed coin-months, prediction rows/positives, and alarm counts where available.

Other fields are preserved in adjacent `*_diagnostics.csv` files, joined using the identifiers. These include alarm precision, base rates, selected thresholds, tuning fields, and calibration details. Frozen run tables are unchanged. `schema.py` defines this layout; it does not recalculate the headline metrics.

AP uses pooled prediction scores. Episode recall uses targetable episodes. Median lead time is conditional on true alarms. Observed calendar coin-months include partial exposure. Blank metrics retain their original undefined meaning.

## Exporters and helpers

| Module | Role and output |
|---|---|
| `model_results` | Combine RF/XGB/LR saved summaries and yearly results for h1/h6/h24 into `reports/model_results/`. |
| `by_coin` | Recompute per-coin metrics from saved predictions/alarms, verify aggregate count agreement, and record source hashes. |
| `calibration_audit` | Read saved fit metadata into `reports/validation_checks/`; primary h6/h24 by default, optional h1. |
| `model_interpretation` | Export coefficients, RF importance, and figures; calculate XGB SHAP rankings from the selected saved fits under `reports/model_interpretation/`. |
| `coverage_table` | Source completeness against each coin's full observed timestamp span under `reports/data_coverage/`. |
| `event_table` | Group episodes into events, add annotations/classifications, and export event tables/figures. Also regenerates `data/processed/depeg_episodes_w_event_type.parquet`; the coin/year heatmap goes under `reports/depeg_episodes/`. |
| `detection_by_type` | Detection counts and conditional lead time by structural event type. False alarms have no observed event type, so this is not a full five-metric comparison. |
| `event_annotations` | Historical context heuristics, not causal labels; helper used by `event_table`. |
| `event_types` | Structural event classifications; helper used by `event_table`. |
| `benchmarks` | Fit constant-prior/persistence benchmarks on saved partitions; separate from saved-model-only reproduction. |
| `schema` | Consistent comparison column order and separation of diagnostics. |

h1 is supplementary; h6/h24 are primary. Benchmark coverage remains h6/h24. Event-type detections cover all three horizons. Figures and interpretation exporters currently cover primary h6/h24.
