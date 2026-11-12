# Report updates: final reviewer organisation

## Final arrangement

Only one reviewer run is retained under `artifacts/final_run/`. It is the
validated run previously called stable-a; its models and scientific results
have not changed. Stable-b established independent repeatability, after which
the duplicate workspaces were removed. The small comparison records remain
under `reports/validation_checks/reproducibility/`.

```text
data/
  raw/                      retained acquisition inputs
  interim/                  derived feature tables
  processed/                final dataset and episode records
artifacts/
  README.md
  index.json
  final_run/
    run.json
    relocation.json
    summary.csv
    by_year.csv
    alarm_events.csv
    episodes.parquet
    fits/<model_tier_horizon_year>/
    source/
reports/
  baseline.json
  model_results/
  model_interpretation/
  robustness/
  validation_checks/
output/
  reproduction/             created by a new full reproduction
```

## Section 6.1: replace the folder descriptions

The repository separates data preparation, model fitting, saved model evidence
and published outputs. The `data/` directory stores source inputs, derived
features and final datasets, while `data_processing/` prepares these inputs for
analysis. The `modelling/` directory defines the models and runs training,
evaluation, robustness checks, reporting and verification. The single final
run is retained in `artifacts/final_run/`, including fitted models, saved data
partitions, prediction scores, alarms and run settings. The tables and figures
used in the report are stored in `reports/`. Reproduction instructions and data
and code descriptions are provided in `docs/`, while `tests/` contains automated
implementation checks. Historical research and earlier report snapshots are
kept separately in `archive/`.

| Folder | Purpose |
|---|---|
| `data/` | Source inputs, derived features and the final modelling datasets. |
| `data_processing/` | Data acquisition, validation, feature calculation, episode labelling and sample preparation. |
| `modelling/` | Model definitions, training, evaluation, robustness checks, report generation and saved-run verification. |
| `artifacts/final_run/` | One final fitted run: models, partitions, prediction scores, alarms, settings and source snapshot. |
| `reports/` | Published tables and figures, verification records and the selected report baseline. |
| `tests/` | Automated checks of the data and modelling implementation. |
| `docs/` | Methodology, data definitions, reproduction instructions and code inventory. |
| `archive/` | Historical experiments, earlier report snapshots and relocation records. |
| `output/` | Fresh reproduction workspaces and review documents; these do not replace the published run. |

The distinction is that `artifacts/final_run/` retains the evidence behind the
fitted models, whereas `reports/` contains the outputs presented to readers.
The final run combines all three model families, both tiers, horizons 1/6/24
and all five test years; there are no separate active stable-a/stable-b folders.

## Section 6.2: data storage

Keep the `raw/`, `interim/` and `processed/` explanation. The market acquisition
table is `data/raw/hourly_stablecoin.parquet`. The canonical final inputs are
now directly under `data/processed/`, including `master.parquet`,
`depeg_episodes.parquet` and `dataset_metadata.json`. Do not describe the final
inputs as living inside a stable-a workspace. They are byte-identical to the
inputs used to train the retained final run. Earlier inputs were moved to the
historical archive and are not used by the current workflow.

## Section 6.3: final output paragraph

The final hourly dataset is stored as `data/processed/master.parquet`, containing
observations, predictors, labels and eligibility indicators. Episode records
are stored in `data/processed/depeg_episodes.parquet`, and construction details
and checksums are recorded in `data/processed/dataset_metadata.json`.
`data/processed/depeg_episodes_w_event_type.parquet` is retained as an annotated
reporting file; fresh full reproductions generate current event classifications
under `reports/depeg_events/` rather than necessarily rewriting this extra
Parquet. The exact dataset used by the final model run is verified against the
input hashes in `artifacts/final_run/run.json`.

Keep the corrections identifying `build_market_features.py` as the market
feature calculator and describing window-local, numerically stable volatility.
Neither feature definitions nor scientific result values changed in this
folder consolidation.

## Section 6.5: replacement reproduction paragraph

The repository supports reproduction from saved model evidence and full
reproduction from retained source inputs. Running `python -m reports.reproduce`
selects the final run in `artifacts/final_run/` through `reports/baseline.json`
and regenerates the published tables and figures without refitting the primary
models. Running `python -m reports.reproduce --full` rebuilds the datasets,
retrains the models and generates fresh outputs under `output/reproduction/`,
leaving the published final run unchanged. Two independent full runs matched
the scientific results before one was retained as the final reviewer baseline;
the comparison records are kept under
`reports/validation_checks/reproducibility/`. Detailed environment setup,
required inputs and commands are provided in `docs/reproduction.md`.

Add a practical availability note: large fitted models and prediction partitions
must be supplied separately for saved-model replay. They are ignored by Git.
A reviewer with the retained raw inputs can instead run full reproduction.
Keep the separate caveat about fresh API acquisition and unconfirmed Dune query
settings. Do not call `--from-processed` a source-data rebuild; it only retrains
from the canonical processed inputs.

## Appendix C1: data table

No stable-a path is needed. Use `raw/`, `interim/` and `processed/` as the
subfolders. Move the hourly market source row to `raw/`. The final processed
pair now refers directly to the canonical paths described above.

## Appendix C4: reports table

Add or revise these rows. The numerical report filenames otherwise stay the same.

| Subfolder | File name | Purpose |
|---|---|---|
| - | `baseline.json` | Selects the final saved run and canonical data root; records explicit historical-input relocations. |
| - | `verification.json` | Records verification of the final run's saved models, partitions, cutoffs, alarms and metrics. |
| - | `manifest.json` | Identifies the published report sources and final dataset metadata. |
| `validation_checks/reproducibility/` | `full_comparison.json` | Records the independent full-workflow comparison. |
| `validation_checks/reproducibility/` | `training_comparison.json` | Records exact agreement in saved training, calibration, prediction and alarm tables. |

## Appendix C5: replace the artifacts introduction and table

The artifacts directory retains one final model run for reviewer inspection.
`artifacts/final_run/` combines logistic regression, random forest and XGBoost
across both feature tiers, horizons 1/6/24 and test years 2021-2025. Each folder
under `fits/` identifies a model, tier, horizon and test year. Its files retain
the fitted models and the observations and scores used to obtain the reported
results. The 6- and 24-hour horizons remain primary; h1 is supplementary.

| Subfolder | File name | Purpose |
|---|---|---|
| - | `README.md` | Explains the final model evidence and how reviewers can use it. |
| - | `index.json` | Identifies the one retained final run and its manifest checksum. |
| `final_run/` | `run.json` | Original training settings, versions, input hashes and completion status. |
| `final_run/` | `relocation.json` | Records the move to reviewer paths without changing the original trained evidence. |
| `final_run/` | `summary.csv` | Pooled results across test years for every model, tier and horizon. |
| `final_run/` | `by_year.csv` | Model performance for each test year. |
| `final_run/` | `alarm_events.csv` | Combined alarm records from the final run. |
| `final_run/` | `episodes.parquet` | Episode definitions used in saved evaluation. |
| `final_run/fits/<fit>/` | `fit.json` | Settings, cutoff, identities, sample sizes and fitted-file checksums for one fit. |
| `final_run/fits/<fit>/` | `practice_pipeline.joblib`, `pipeline.joblib` | Practice and final fitted models, including fitted preprocessing. |
| `final_run/fits/<fit>/` | `inner_train.parquet`, `inner_val.parquet`, `train.parquet`, `test.parquet` | Saved observations for each stage; calibration/test files also contain scores and cutoffs. |
| `final_run/fits/<fit>/` | `alarms.csv` | Grouped alarms for that model, tier, horizon and test year. |
| `final_run/fits/<logistic fit>/` | `tuning.json`, `cv/`, `cv_skipped.json` | Logistic-regression tuning results and calendar-validation evidence. |
| `final_run/source/` | Original Python source files | Immutable code snapshot captured during training. |

Remove the old active-run layout under `h1/` and `h6_and_h24/`. Do not put
stable-a and stable-b in the final artifacts table. SHAP rankings and figures
belong in the report table under `reports/model_interpretation/xgboost/`, not
in a superseded run-root `shap_*` section.

## Appendix C6-C8

Keep `test_numerical_reproducibility.py` in the tests inventory. The artifact
utility tests now also check explicit input relocation; workflow tests check
published-run selection. Add this code-organisation note to the documentation
inventory if it is intended to list every document. The archive table should omit superseded report snapshots, which were removed.
Promotion and relocation checksum records are retained under
`reports/validation_checks/reproducibility/`.
Do not describe discarded temporary model runs as retained reviewer evidence.

## What does not change

There are no new performance numbers from this reorganisation. LR/RF/XGBoost
results, feature attributions and robustness tables remain the already validated
published baseline. The changes in the earlier proposal revision report still
apply; this note supersedes its temporary workspace paths and artifacts layout.
