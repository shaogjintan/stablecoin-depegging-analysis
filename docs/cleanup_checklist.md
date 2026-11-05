# Cleanup record

Implemented with no permanent deletions. docs/cleanup_moves.json records the main relocations. Original modified reporting/helper sources are preserved under archive/history/pre_cleanup_source/. Git's pre-cleanup staged checkpoint remains unchanged; cleanup changes are not staged or committed.

## Completed

- Historical folders, planning documents, earlier reports and alternative runners relocated under archive/history/.
- Deletion candidates quarantined under delete/, including CSV exports. The superseded robustness results were restored from the earlier temporary backup into delete/.
- Original dataset source and acquired hourly table moved to snapshots/; feature tables moved to interim/. Corrected input hashes retained.
- All six retained runs moved to primary/supplementary/sensitivity groups; earlier non-refit LR run archived.
- Shared LR calendar CV, fitting, matched sample, formatting and alternative-episode helpers extracted before retiring old runners.
- Historical-context annotations retained separately from structural event types.
- Corrected event/type/detection and consolidated coverage tables generated.
- Saved-final LR coefficients, RF importance and XGB SHAP interpretation figures generated without retraining.
- Existing presentation CSV tables reproduce with exactly identical parsed values; all three robustness tables reproduce byte-for-byte from relocated evidence.
- Current README, methodology, reproduction guide, data dictionary, results guide and artifact instructions added.
- Rebuilding corrected tables in memory from the relocated original snapshot matches stored values and dtypes (normalising the non-persisted index).
- Local CPython 3.12 environment frozen; package metadata and optional acquisition dependencies declared.
- Data/run manifests and independent final-run verifier added; all 90 final-run cells pass the saved-alarm/metric verification scope.
- Large relocated artifacts excluded from ordinary Git staging; dune_onchain.zip kept.

## Remaining release tasks and limitations

- Run artifacts remain local; publish a checksum-backed artifact package or provide a controlled download before claiming a fresh clone can replay saved results. No artifact upload or retraining was performed during cleanup.
- The environment lock was checked locally; installation on a fresh machine and Windows remains to be verified.
- Historical annotations still require human report review; they are context, not established causes.
- CSV exports in delete/ are preserved, not certified equivalent or permanently deleted.
- Both sensitivity runs are retained with their original verification evidence, but have not been independently reverified by the new final-refit verifier.
- The default final-run verifier checks hashes, saved alarms and metrics; full model-score replay requires --check-model-scores. It does not rerun LR C selection. Threshold selection uses the maintained exhaustive engine, with independent synthetic oracle tests.
- New LR/RF interpretation figures cover both primary horizons; XGB figures reuse existing final SHAP rankings rather than recomputing SHAP.
- Six independent LR/calendar-CV/preprocessing/threshold tests were migrated to the maintained suite; protocol-specific tests and original verifiers are preserved in archive/history/; further portability/hardening checks can be migrated deliberately.
