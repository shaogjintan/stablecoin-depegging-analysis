# Stable reproduction comparison

Primary training in `artifacts/reproduction-stable-a/` and
`artifacts/reproduction-stable-b/` matches exactly across all 90 cells.
Both saved audits passed, including model-score replay.

The exact comparison covers 455 tables: the master and episode datasets;
all four saved partitions and alarm tables for every fit; and the combined
summary, yearly results and alarms. Scientific fit metadata also matches,
excluding run IDs and pipeline checksums. There are 18 pooled summary rows,
90 yearly rows and 6,918 combined alarm records. No numerical tolerance was
used for this between-run comparison.

This establishes reproducibility of the primary data, training, calibration,
prediction and alarm results in the tested environment. Other software versions
and hardware have not been tested.

Both complete workflows have now finished all 16 stages. The full comparison
passed 490 checks, including the generated report CSVs. Two interpretation
tables contain different run IDs, as expected: each ID was checked against its
own run manifest and all remaining values match exactly. Workspace paths were
normalised. Figures and path-bearing JSON manifests were not compared bytewise.

## Changes against the original report baseline

LR's pooled headline metrics are unchanged. Episode recall is unchanged for
all models. RF false-alarm counts differ in two cells, by at most two alarms;
XGBoost counts differ in six cells, by at most 13. RF/XGBoost AP differ in six
cells each. Two XGBoost median lead times differ, by at most one hour.
The original report baseline has now been archived, and `reports/` has been
updated to stable-a.

All 30 changed pooled headline values are listed below. Only the comparison
against the old baseline uses rtol=1e-10 and atol=1e-12. Stable-a and stable-b
values are exactly equal.

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

## Evidence

Local training comparison: `reports/validation_checks/reproducibility/training_comparison.json`.
Automatic full comparison: `reports/validation_checks/reproducibility/full_comparison.json`.
Local diagnostic artifacts are ignored in Git.

## Publication of the stable baseline

The report baseline has been updated to stable-a, selected in
`reports/baseline.json`. All 29 current report CSVs match stable-a exactly.
Superseded report and input snapshots were removed after validation.
Old/new SHA-256 hashes remain in
`reports/validation_checks/reproducibility/report_promotion.json`; input
relocation checksums remain in `reviewer_relocation.json` in the same folder.

Fifteen existing CSVs were updated and six alarm-rule CSVs were added. Related
interpretation figures, source manifests and verification records were updated
together. Unchanged coverage, event and benchmark table values were preserved.
The complete full-workflow scope now includes h1 in the calibration audit
(60 to 90 rows), LR coefficients (360 to 540) and RF importance (320 to 480).
LR's h6/h24 coefficient values and all LR headline performance values remain
unchanged; the larger coefficient table adds supplementary h1 records.

`python -m reports.reproduce` selects this published run and its actual data
workspace, verifies its identity and regenerates reports without training.
Direct report exporters use the selected run where applicable. The original
`data/processed/` pair remains available for historical comparison; its values
are not silently substituted for the published stable-a training inputs.

Validation: 82 automated tests passed, all 29 published CSVs matched stable-a
exactly, promoted output checksums matched their source files, and archived
file checksums matched the pre-promotion reports. Figures were copied from
stable-a; the independent a/b comparison checked scientific CSV values rather
than image bytes.


## Superseded workspace cleanup

The project owner subsequently requested deletion of the superseded artifact
folders: `artifacts/h1/`, `artifacts/h6_and_h24/` and the initial
`artifacts/reproduction/`. Their historical paths above describe the evidence
reviewed before removal; the old fitted models are no longer available there.
Stable-a, stable-b, the original report snapshot and the small comparison
summaries remain retained. The two final comparison JSON summaries are tracked
in Git; other local diagnostic artifacts remain ignored.

## Current reviewer layout (12 November 2026)

The selected validated run was moved, without changing model/result values, to
`artifacts/final_run/`. Its exact datasets are now under `data/processed/`.
The duplicate stable-a/stable-b workspaces and large diagnostic trials were
removed after verification; the names above describe the historical comparison,
not current reviewer paths. Small final audit records are retained under
`reports/validation_checks/reproducibility/`. Original run metadata remains
immutable, and its historical input paths are translated by the explicit
relocation map in `reports/baseline.json`. See
`docs/reviewer_code_organisation.md` for the final folder table and report edits.
