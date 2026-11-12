# Final model evidence for reviewers

`final_run/` contains the one validated run used by the published report.
It covers logistic regression, random forest and XGBoost, both feature tiers,
horizons 1/6/24 and test years 2021-2025. Horizons 6 and 24 are primary; h1 is
supplementary. Two independent full reproductions agreed before their temporary
workspaces were removed. Their small comparison records are retained under
`reports/validation_checks/reproducibility/`.

| Location | Contents |
|---|---|
| `index.json` | Identity, status and manifest checksum of the final run. |
| `final_run/run.json` | Original training settings, software versions and input hashes. |
| `final_run/relocation.json` | Mapping from original execution paths to reviewer locations. |
| `final_run/summary.csv` | Pooled model results. |
| `final_run/by_year.csv` | Test-year results. |
| `final_run/alarm_events.csv` | Combined alarm records. |
| `final_run/episodes.parquet` | Episodes used by the saved evaluation. |
| `final_run/fits/<fit>/` | Fitted models, partitions, predictions, alarms and fit metadata. |
| `final_run/source/` | Code snapshot captured when the run was trained. |

Each fit identifies a model, tier, horizon and test year. It contains
`practice_pipeline.joblib`, `pipeline.joblib`, `fit.json`, `inner_train.parquet`,
`inner_val.parquet`, `train.parquet`, `test.parquet` and `alarms.csv`.
Logistic fits also retain `tuning.json`, `cv/` and skipped-fold diagnostics.

The canonical dataset is under `data/processed/`. The publication tables and
figures are under `reports/`; `reports/baseline.json` selects this run and its
data. Original run metadata and fitted files remain unchanged. Historical input
paths in `run.json` resolve through an explicit checksum-backed relocation map.

Large fitted models, partition Parquets and source snapshots are ignored by Git
and must be supplied separately for saved-model replay. Small result records
are versioned. A reviewer without the large evidence can rebuild and retrain
from retained raw inputs using `python -m reports.reproduce --full`.
New reproduction workspaces go to `output/reproduction/`, leaving this final
run unchanged. The duplicate validation workspaces are not needed for review.
