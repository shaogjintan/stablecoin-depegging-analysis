# Final run evidence

Only the four final refit runs remain active:

| Horizons | Models | Directory under artifacts/ |
|---|---|---|
| h6/h24 | RF and XGBoost | h6_and_h24/random_forest_xgboost/ |
| h6/h24 | Logistic regression | h6_and_h24/logistic_regression/ |
| h1 | RF and XGBoost | h1/random_forest_xgboost/ |
| h1 | Logistic regression | h1/logistic_regression/ |

Each directory now contains run.json, summaries, alarms, fits/, episodes.parquet,
and its original source snapshot. All relocated files were verified byte-for-byte
unchanged. Original paths within frozen evidence are retained for provenance;
docs/cleanup_moves.json records relocations. index.json lists final run identities.

The older no-refit experiments and evidence are archived under
archive/history/artifacts/runs/{sensitivity,sensitivity_checks}/. Their historical
index and original verification status are retained there. They are not used by
the maintained final-result exporters or outcome-definition robustness checks.

Large models, prediction Parquets and source snapshots are ignored by Git. A
clone alone is insufficient for saved-run verification. No public model-artifact
download has been published. Supply these final run folders separately, or
retrain from the retained sources using python -m reports.reproduce --full.
Source-data distribution bundles remain under artifacts/distribution/.

Run CSVs are original per-run outputs used by the exporters.
reports/model_results/ holds their combined reader-facing tables; fitted
models, partition predictions, and source snapshots remain here as evidence.
