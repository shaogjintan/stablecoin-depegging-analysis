# Modelling pipeline

Run package commands from the repository root. For each test fold, a practice
model scores the calibration year and selects an alarm cutoff. The final model
refits on the full eligible training period and transfers that cutoff unchanged
to the test year. Horizons 6 and 24 are primary; h1 is supplementary.

| Folder | Responsibility |
|---|---|
| `models/` | Estimator classes and model-specific preprocessing. |
| `training/` | Training runner, calendar validation, logistic tuning and fixed parameters. |
| `evaluation/` | Grouped alarms, targetability, five headline metrics and cutoff selection. |
| `robustness_checks/` | Alternative episode definitions, alarm smoothing and model-pair OR rules. |
| `validation/` | Verification of saved models, partitions, scores, alarms and metrics. |
| `reporting/` | Performance tables, interpretation, calibration audits, coverage and event outputs. |
| `utils/` | Checksums, JSON writing and saved-evidence resolution. |

Feature definitions, sample preparation, matched populations and dataset
construction live in `data_processing/`. Model classes retain their module
paths so saved fitted models can be loaded.

The published evidence is `artifacts/final_run/`, its exact inputs are in
`data/processed/`, and tables and figures are in `reports/`.

## Commands

```sh
# Validate retained data without replacing it.
python -m data_processing.build_dataset

# Run automated implementation checks.
make test

# Verify saved model evidence, including score replay.
python -m modelling.validation.verify_model_results --check-model-scores

# Regenerate published reports from the saved final run.
python -m reports.reproduce

# Export calibration records for all three horizons.
python -m modelling.reporting.calibration_audit --include-h1

# Train a new run in a fresh directory.
python -m modelling.training.train_models --out-dir output/new-model-run
```

The training runner defaults to all three models and horizons 1/6/24. Use
`--models`, `--horizon` or `--year` to select a smaller scope. Fixed settings
and the logistic C grid are in `training/parameters.py`. Chronological CV
splits are in `training/calendar_cv.py`; logistic tuning is in
`training/logistic_tuning.py`.

`make test` checks implementation behaviour. Saved-run verification checks the
retained research evidence; its own behaviour is covered by automated tests.
Benchmark exporters fit simple constant-prior and persistence comparisons on
saved partitions without refitting the primary models.

See [reproduction instructions](../docs/reproduction.md) for the complete
source-data rebuild and training workflow. Archived code is excluded from that
workflow.
