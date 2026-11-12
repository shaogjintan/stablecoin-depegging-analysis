# Repository organisation

The repository separates data preparation, model fitting, saved model evidence
and published outputs. Run commands from the repository root. Detailed commands
are in [reproduction](reproduction.md), and maintained Python files are listed
in [the code inventory](code_inventory.md).

| Folder | Purpose |
|---|---|
| `data/` | Retained source inputs, derived features and final modelling datasets. |
| `data_processing/` | Acquisition, validation, feature calculation, episode labels, eligibility and sample preparation. |
| `modelling/` | Models, training, evaluation, robustness checks, reporting and saved-run verification. |
| `artifacts/final_run/` | One published fitted run, including models, partitions, scores, alarms and source snapshot. |
| `reports/` | Published tables, figures and verification records; also contains the reproduction entry point. |
| `tests/` | Automated checks of data processing and modelling behaviour. |
| `docs/` | Methodology, data definitions, reproduction instructions and code inventory. |
| `archive/` | Historical experiments and research records, excluded from the maintained workflow. |
| `delete/` | Quarantined obsolete files, excluded from the maintained workflow. |
| `output/` | Fresh reproduction workspaces and locally generated documents. |

## Data storage

`data/raw/` contains retained acquisition inputs, `data/interim/` contains derived
feature tables, and `data/processed/` contains the final dataset and episodes.
The acquired hourly market table is `data/raw/hourly_stablecoin.parquet`.
The exact model inputs are `data/processed/master.parquet` and
`data/processed/depeg_episodes.parquet`. Dataset metadata records construction
settings; `data/manifest.json` records paths, schemas and hashes.

`data/processed/depeg_episodes_w_event_type.parquet` is regenerated alongside
the event report tables. It preserves episode records and adds event IDs,
structural event types and separate historical-context annotations. Full
reproduction writes it under the new workspace's `data/processed/` directory.

## Published report evidence

| Subfolder | File name | Purpose |
|---|---|---|
| `reports/` | `baseline.json` | Selects the final run and canonical input locations. |
| `reports/` | `manifest.json` | Identifies report sources and dataset metadata. |
| `reports/` | `verification.json` | Checks models, partitions, thresholds, alarms and metrics. |
| `reports/validation_checks/reproducibility/` | `training_comparison.json`, `full_comparison.json` | Records independent reproduction comparisons. |
| `reports/validation_checks/reproducibility/` | `report_promotion.json`, `reviewer_relocation.json` | Retains checksums for baseline replacement and input relocation. |

## Saved models and predictions

The single final run combines all three model families, both feature tiers,
horizons 1/6/24 and five test years. Horizons 6 and 24 are primary; h1 is
supplementary. Large fitted models, partitions and source snapshots must be
supplied separately for saved-model replay; they are ignored by Git.

| Subfolder within `artifacts/` | File name | Purpose |
|---|---|---|
| - | `README.md`, `index.json` | Explains and identifies the published run. |
| `final_run/` | `run.json` | Original settings, software versions and input hashes. |
| `final_run/` | `relocation.json` | Maps original execution paths to reviewer locations. |
| `final_run/` | `summary.csv`, `by_year.csv` | Pooled and test-year performance. |
| `final_run/` | `alarm_events.csv`, `episodes.parquet` | Combined alarms and evaluation episodes. |
| `final_run/fits/<fit>/` | `fit.json` | One fit's model, tier, horizon, year, threshold and checksums. |
| `final_run/fits/<fit>/` | `practice_pipeline.joblib`, `pipeline.joblib` | Practice and final fitted models with preprocessing. |
| `final_run/fits/<fit>/` | `inner_train.parquet`, `inner_val.parquet`, `train.parquet`, `test.parquet` | Stage-specific observations; calibration and test tables include scores. |
| `final_run/fits/<fit>/` | `alarms.csv` | Grouped alarms for this fit. |
| `final_run/fits/<logistic fit>/` | `tuning.json`, `cv/`, `cv_skipped.json` | Logistic calendar-validation and tuning evidence. |
| `final_run/source/` | Python source files | Immutable training-time code snapshot. |

## Reproduction

`python -m reports.reproduce` verifies saved model evidence and regenerates
published reports without refitting the primary models. It separately evaluates
simple benchmarks. `python -m reports.reproduce --full` rebuilds data, retrains
models and writes fresh results under `output/reproduction/`. This uses retained
local sources and leaves the published run unchanged. Fresh API acquisition is
separate and may produce different observations.
