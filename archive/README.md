# Compact research history

This archive is historical context, not an executable reproduction pipeline.
Current code lives in data_processing/ and modelling/, final reports in reports/,
and final run evidence in artifacts/h1/ and artifacts/h6_and_h24/.

Retained material:

- experiment_records/inventory.json: tracks 58 checksummed historical records,
  retained under experiment_records/ and artifacts/. Original paths and hashes
  are preserved.
- artifacts/sensitivity/: descriptive random_forest_xgboost_no_refit/,
  xgboost_no_refit/, and logistic_regression_no_refit/ experiment folders.
  Manifests and original runners/source live with these records.
- Original historical training sources are stored with their experiment
  artifacts. Superseded standalone runners and refactored copies are described
  in experiment_records/historical_training_and_threshold_analysis.md; their
  files remain in delete/archive_cleanup/history/.
- Superseded diagnostics and alarm experiments are described in
  experiment_records/obsolete_data_processing_checks.md and
  experiment_records/abandoned_alarm_experiments.md; their code is quarantined.
- planning/: four original methodology/proposal and supervisor-feedback notes.
  Its README distinguishes historical proposals from the final protocol.
- documentation/requirements*.txt: historical environment records.
- literature_replication/ and midterm_presentation/:
  earlier academic work, distinct from the final analysis.

Superseded models, predictions, datasets, reports, tests and editing checkpoints
have moved to delete/archive_cleanup/history/. No files were permanently deleted.
The original large experiment folders remain intact there; selected compact
records were copied here before relocation and checksum-verified.

Historical records retain their original path strings and conclusions. They
must not be presented as the current protocol or final results. Consult
maintenance/cleanup_moves.json for the full relocation history.

Superseded LR training and exploratory threshold analyses are described in
experiment_records/historical_training_and_threshold_analysis.md. Their code
is preserved under delete/archive_cleanup/history/.

Literature replication retains its original data, result tables, scripts and
vector chart. Midterm material retains its code, tables and notes. Redundant
PNG/figure exports are preserved under delete/archive_cleanup/history/.
Planning filenames now describe their topics; see planning/README.md.
