# Saved models and predictions

`final_run/` contains the one validated run used by the published report.
It covers logistic regression, random forest and XGBoost, both feature tiers,
horizons 1/6/24 and test years 2021-2025. Horizons 6 and 24 are primary; h1 is
supplementary.

| Location | Contents |
|---|---|
| `index.json` | Identity, status and manifest checksum of the final run. |
| `final_run/run.json` | Original training settings, software versions and input hashes. |
| `final_run/input_paths.json` | Mapping from original execution paths to current file locations. |
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
are versioned. Without the saved model files, you can rebuild and retrain
from retained raw inputs using `python -m reports.reproduce --full`.
New reproduction workspaces go to `output/reproduction/`, leaving this final
run unchanged.

## File availability

`final_run/fits/` and `final_run/source/` exist in the local working copy but
are ignored by Git. They will not appear in a GitHub clone unless supplied
separately. `<fit>` is a placeholder for an actual model/tier/horizon/year folder.
The source folder holds an immutable training-time snapshot, not maintained code.

`reports/model_results/` contains readable summaries. This folder additionally
holds the models and predictions needed to verify and regenerate those summaries.
