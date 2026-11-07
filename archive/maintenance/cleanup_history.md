# Cleanup record

Implemented with no permanent deletions. docs/cleanup_moves.json records the main relocations. Compact historical records are preserved under archive/experiment_records/. Redundant source checkpoints and large old outputs are quarantined under delete/archive_cleanup/history/. Consult Git status for the current staging state.

## Completed

- Historical folders, planning documents, earlier reports and alternative runners relocated under archive/.
- Deletion candidates quarantined under delete/, including CSV exports. The superseded robustness results were restored from the earlier temporary backup into delete/.
- Original dataset source and acquired hourly table moved to snapshots/; feature tables moved to interim/. Corrected input hashes retained.
- Four final runs consolidated under artifacts/; older no-refit runs and their evidence archived under archive/artifacts/sensitivity/.
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
- Both historical no-refit runs are archived with original verification evidence; the final-refit verifier does not audit those older protocols.
- The default final-run verifier checks hashes, saved alarms and metrics; full model-score replay requires --check-model-scores. It does not rerun LR C selection. Threshold selection uses the maintained exhaustive engine, with independent synthetic oracle tests.
- New LR/RF interpretation figures cover both primary horizons; XGB figures reuse existing final SHAP rankings rather than recomputing SHAP.
- Six independent LR/calendar-CV/preprocessing/threshold tests were migrated to the maintained suite; protocol-specific tests and original verifiers are preserved in archive/; further portability/hardening checks can be migrated deliberately.

## Canonical dataset follow-up

The old original master and old episode manifest are now quarantined under delete/. The maintained build no longer depends on them. The canonical corrected pair is data/processed/master.parquet and depeg_episodes.parquet. Both build entry points merge source/interim inputs and apply corrections in memory. Existing frozen bytes are preserved. Historical input filenames resolve through a hash-checked alias map. Direct rebuilding was verified against the retained pair: all other columns and all episode data match exactly; rolling volatility differs by less than 1e-9. See reproduction.md for current commands and comparison scope.

## Processing consistency follow-up

The shared builder now lives at data_processing/build_dataset.py; the original modelling command remains a compatibility wrapper. Imports are package-qualified, paths are centralised in config, and writers create their output directories. Configuration imports do not create directories. Acquisition creates its optional HTTP session lazily. Processing modules use 80-column, four-space formatting with grouped imports and refreshed docstrings; .style.yapf records the Google-based convention. All 29 maintained tests pass. A fresh build matches the frozen canonical pair within its documented tolerance, and both canonical file hashes are unchanged.

- Archive reduced to a compact historical record; redundant material remains intact under delete/archive_cleanup/history/. No files were permanently deleted.

- Retained history flattened directly under archive/. delete/ dependency audit passed 72 tests, a real-source preprocessing rebuild, and six sampled saved-model replays while quarantine was unavailable. Scope: docs/delete_review.md.
