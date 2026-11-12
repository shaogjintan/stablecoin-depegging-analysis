# Reproduction comparison and result changes

The published run is `artifacts/final_run/`, with exact input datasets under
`data/processed/` and report outputs under `reports/`. `reports/baseline.json`
selects this evidence. Two independent full reproductions matched before one
run was retained and the duplicate workspaces were removed.

## Comparison scope

Both workflows completed all 16 stages and passed their 90-cell saved-model
score audits. Primary comparison covered 455 tables: the master and episode
datasets, four partitions and alarms for each fit, and combined summary,
yearly and alarm tables. Scientific fit metadata also matched, excluding run
identities and serialized-model checksums. These tables contain 18 pooled
results, 90 yearly results and 6,918 combined alarms.

The full comparison passed 490 checks, including downstream scientific CSVs.
Expected run identifiers were validated against each run's manifest before
comparison; workspace paths were normalized. Scientific values matched exactly.
Figures and path-bearing JSON files were not compared bytewise. This evidence
applies to the tested software environment, not arbitrary platforms.

## Changes from the superseded baseline

The numerical corrections are explained in
[numerical reproducibility](reproducibility_investigation.md). LR pooled headline
metrics and episode recall for every model were unchanged. RF false-alarm
counts changed in two cells; XGBoost counts changed in six. RF/XGBoost average
precision changed in all six pooled cells each. Two XGBoost median lead times
changed. The table records all 30 changed pooled headline values; comparisons
against the superseded baseline used rtol=1e-10 and atol=1e-12. The two new
reproductions matched exactly.

| Model | Tier | Horizon | Metric | Original | Stable |
|---|---|---:|---|---:|---:|
| random_forest | 1 | 1 | average_precision | 0.00217456564015 | 0.00217453899741 |
| random_forest | 1 | 6 | average_precision | 0.00520092398515 | 0.00520201655959 |
| random_forest | 1 | 24 | average_precision | 0.0179589076032 | 0.0179567035878 |
| random_forest | 2 | 1 | false_alarms | 292 | 294 |
| random_forest | 2 | 1 | false_alarms_per_coin_month | 0.924050632911 | 0.930379746835 |
| random_forest | 2 | 1 | average_precision | 0.000904236160064 | 0.000905699096363 |
| random_forest | 2 | 6 | average_precision | 0.00486776721872 | 0.0048690951173 |
| random_forest | 2 | 24 | false_alarms | 184 | 183 |
| random_forest | 2 | 24 | false_alarms_per_coin_month | 0.582278481013 | 0.579113924051 |
| random_forest | 2 | 24 | average_precision | 0.0172888112548 | 0.0172895855874 |
| xgboost | 1 | 1 | false_alarms | 631 | 644 |
| xgboost | 1 | 1 | false_alarms_per_coin_month | 1.99683544304 | 2.03797468354 |
| xgboost | 1 | 1 | average_precision | 0.000964727461157 | 0.00101507341662 |
| xgboost | 1 | 6 | median_lead_hours | 2.5 | 3 |
| xgboost | 1 | 6 | false_alarms | 303 | 300 |
| xgboost | 1 | 6 | false_alarms_per_coin_month | 0.958860759494 | 0.949367088608 |
| xgboost | 1 | 6 | average_precision | 0.00526392352184 | 0.00560959408769 |
| xgboost | 1 | 24 | median_lead_hours | 5.5 | 6.5 |
| xgboost | 1 | 24 | false_alarms | 296 | 301 |
| xgboost | 1 | 24 | false_alarms_per_coin_month | 0.936708860759 | 0.95253164557 |
| xgboost | 1 | 24 | average_precision | 0.0148114990661 | 0.0143845830578 |
| xgboost | 2 | 1 | false_alarms | 766 | 764 |
| xgboost | 2 | 1 | false_alarms_per_coin_month | 2.42405063291 | 2.41772151899 |
| xgboost | 2 | 1 | average_precision | 0.00145768446359 | 0.00128914227707 |
| xgboost | 2 | 6 | false_alarms | 421 | 423 |
| xgboost | 2 | 6 | false_alarms_per_coin_month | 1.33227848101 | 1.33860759494 |
| xgboost | 2 | 6 | average_precision | 0.00438686346244 | 0.00429688570265 |
| xgboost | 2 | 24 | false_alarms | 245 | 238 |
| xgboost | 2 | 24 | false_alarms_per_coin_month | 0.775316455696 | 0.753164556962 |
| xgboost | 2 | 24 | average_precision | 0.0163145158362 | 0.0164386473936 |


## Retained evidence

| Record | Purpose |
|---|---|
| `reports/validation_checks/reproducibility/training_comparison.json` | Exact primary training and result comparison. |
| `reports/validation_checks/reproducibility/full_comparison.json` | Full workflow comparison, including report tables. |
| `reports/validation_checks/reproducibility/report_promotion.json` | Checksums recording replacement of the superseded reports. |
| `reports/validation_checks/reproducibility/reviewer_relocation.json` | Checksums and relocation of the canonical inputs. |
| `reports/verification.json` | Verification of the selected published model evidence. |

Superseded inputs and report snapshots were removed after validation. Their
former paths in audit records describe the evidence checked at that stage;
they are not current reproduction inputs. Current model-result tables,
interpretation outputs and robustness reports use the selected final run.
