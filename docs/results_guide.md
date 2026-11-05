# Results guide

| Report claim or exhibit | Authoritative output |
|---|---|
| Model performance, matched tiers, h6/h24 | reports/final/presentation/model_comparison_summary.csv |
| Test-year performance | reports/final/presentation/model_comparison_by_year.csv |
| Validation exposure, cutoff and false alarms | reports/final/presentation/calibration_audit.csv |
| Constant-prior and persistence comparison | reports/final/presentation/benchmark_comparison.csv |
| Effect of final refit | reports/final/presentation/refit_vs_no_refit.csv |
| Alternative episode definitions | reports/final/robustness/metrics.csv |
| Tier tradeoffs under alternatives | reports/final/robustness/onchain_effects.csv |
| Grouped events and historical context | reports/final/events/episodes.csv |
| Structural event types | reports/final/events/types_events.csv |
| Final detections by type (counts) | reports/final/events/caught_by_type.csv |
| Source completeness with explicit denominator | reports/final/coverage/coverage.csv |
| Final LR coefficients | reports/final/interpretation/lr_coefficients.csv |
| Final RF importance | reports/final/interpretation/rf_importance_by_year.csv |
| Final XGB SHAP ranking/figures | reports/final/interpretation/ |

The h1 results are supplementary, retained under artifacts/runs/supplementary/. The primary presentation exporter covers h6/h24. Counts by type also cover h1. Historical annotations must be reviewed as context; date membership is not evidence of causation.

All old final_results/, fixed/tuned figures and non-refit coefficients are archived. Do not combine their values with the current final-model outputs. Frozen JSON manifests may contain old paths; resolve them with docs/cleanup_moves.json rather than editing the original provenance.
