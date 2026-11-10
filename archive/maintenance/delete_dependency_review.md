# Delete-folder dependency review

The retained archive now lives directly under archive/. delete/ contains
quarantined older datasets, model evidence, generated reports, superseded code
and editing checkpoints. No files were permanently deleted during this review.

## Checks performed

The delete/ directory was temporarily renamed out of its expected location and
restored unchanged afterward. While it was unavailable:

- All 72 maintained tests passed.
- Both interim feature tables and the processed dataset were rebuilt from the
  retained hourly market source and Dune ZIP in a temporary workspace.
- Rebuilt model inputs matched the frozen comparison scope: historical
  completeness/simple-return diagnostics may differ, and the documented
  rolling-volatility tolerance applies. The episode manifest matched exactly.
- Frozen processed input hashes were unchanged.
- Saved predictions were replayed for six pipelines spanning LR, RF and XGBoost
  across all four final runs, using up to 128 saved test rows per pipeline.
  Predicted scores matched the saved scores within the verifier's tolerance.

Maintained code was searched for archive/delete dependencies. Report/data
provenance references no required input under delete/. The
previous_preparation_metadata field in data/processed/dataset_metadata.json is
an informational link to an older metadata record, not a runtime input.
Documentation and the relocation map also contain historical quarantine paths.

## Scope

No runtime dependency on delete/ was found in these checks. Full primary model
retraining and the complete report-generation workflow were not rerun. This
review does not claim every possible optional/historical command was executed.
Deleting quarantine material later would discard history; it was not performed
or required to establish the supported workflow's independence.
