# Modelling pipeline

Run package commands from the repository root. The primary workflow uses a practice model to calibrate an alarm cutoff, then refits on the full eligible training window and transfers that cutoff unchanged to the test year. The primary horizons are h6/h24; h1 is supplementary.

| Folder | Responsibility |
|---|---|
| `models/` | Estimator classes and preprocessing used by saved model files. Keep these module paths stable. |
| `training/` | Main refit runner, training-only calendar CV, logistic fitting, and fixed estimator parameters. |
| `evaluation/` | Grouped alarms, targetability, metrics, exhaustive threshold selection, and pooled run summaries. |
| `robustness_checks/` | Alternative episode definitions evaluated using frozen predictions and alarms. |
| `validation/` | Independent checks of saved run evidence and preprocessing. |
| `reporting/` | CSV/figure exporters, calibration audit, and benchmark comparisons. |
| `utils/` | Checksums, JSON writing, and relocated saved-evidence resolution. |

Feature definitions, sample preparation, matched populations, dataset transformations, and canonical dataset construction live in `data_processing/`. See `data_processing/dataset_transformation.py` and `data_processing/build_dataset.py`.

## Main commands

```sh
# Verify retained inputs without rebuilding them.
python -m data_processing.build_dataset

# Run automated implementation checks.
make test

# Verify saved run evidence without training.
python -m modelling.validation.verify_model_results

# Regenerate reports from retained inputs and saved models.
python -m reports.reproduce

# Generate the primary h6/h24 calibration audit without training.
python -m modelling.reporting.calibration_audit

# Train new models into a fresh directory only.
python -m modelling.training.train_models --out-dir artifacts/runs/primary/ALL-MODELS-REPRODUCED --models logreg random_forest xgboost --horizon 6 --horizon 24
```

`reporting.benchmarks` fits constant-prior and persistence benchmarks on retained partitions; it is separate from saved-model-only report reproduction. It does not retrain primary RF/XGB/LR models. Obsolete slide support and Markdown formatting are archived under `delete/archive_cleanup/history/modelling_redundant/reporting/`.

`evaluation.threshold_selection` implements the final coin-month budget policy. Unused generic hour-level curves, F1/F-beta threshold selection, and the old standalone evaluation exporter are archived in `delete/archive_cleanup/history/modelling_redundant/evaluation/legacy_evaluation.py`. `evaluation.metrics` contains only the maintained grouped-alarm metrics.

Historical launchers and obsolete compatibility shims are preserved under `delete/archive_cleanup/history/`; original experiment sources remain with their historical artifacts. Current commands and imports use the folders above. Original run manifests and source snapshots remain unchanged; their older module paths describe the historical runs. Relocations are recorded in `archive/maintenance/cleanup_moves.json`.

Historical RF/XGBoost no-refit runners are archived under `archive/artifacts/sensitivity/`. Saved no-refit summaries remain in archive/artifacts/sensitivity/; the maintained sensitivity workflow uses alternative outcome definitions only.

Training settings live in `training/parameters.py`: `XGBOOST_FIXED_PARAMS`, `RANDOM_FOREST_FIXED_PARAMS`, and `DEFAULT_C_GRID`. Calendar CV only constructs chronological splits; logistic fitting and C selection live in `training/logistic_tuning.py`. The main runner defaults to RF, XGBoost, and logistic regression across all three horizons (h1/h6/h24). Use `--models` to select a subset.

The audit implementation under `validation/` is itself tested by `tests/test_saved_run_verification.py`. `make test` checks code behaviour; `make verify` checks the retained research evidence, including all five headline metrics. Neither is a substitute for the other.

Shared utilities stream SHA-256 checksums in 1 MiB chunks. Evidence lookup checks every fit run ID across the retained two-level run layout, rejects mixed/incomplete or ambiguous evidence, requires the episode snapshot, and does not cache paths. Fit-cell scope and result correctness are checked separately by the verifier. Tests for these utilities live in `tests/test_artifact_utilities.py`.
