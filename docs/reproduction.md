# Reproduction

Run commands from the repository root using CPython 3.12. `requirements.lock` records all packages in the local macOS environment used for cleanup validation; it is an environment freeze, not a wheel/hash lock. The original Windows requirements are preserved under archive/history/documentation/. Cross-platform numerical differences should be assessed with tolerances rather than assuming byte-identical model files.

## Frozen input validation

Both entry points use the same direct build path:

```sh
python -m data_processing.build_dataset
python -m data_processing.merge_and_label
```

They validate the existing canonical `data/processed/master.parquet` and `depeg_episodes.parquet`. If missing, they merge the acquired hourly snapshot and two interim feature tables, then apply corrected features/labels entirely in memory and write only the corrected canonical pair plus metadata. No original master or legacy episode file is needed or produced.

Add `--rebuild` to rebuild and compare with existing files without replacing them; use `--rebuild --out-dir <fresh-directory>` to write a new pair. Incomplete pairs fail explicitly. Already-corrected input is rejected to prevent shifting clocks twice. The build preserves labels, episode definitions and sample populations. Source-block layout can change rolling standard deviations by up to 1e-9 in absolute value near zero; the comparison checks those two columns with that tolerance and requires every other column to match exactly. Existing frozen canonical bytes and their model results remain unchanged. A freshly rebuilt file may have a different checksum and should not be silently substituted for frozen model evidence.

Historical manifest filenames resolve to the canonical pair through an explicit alias map; hashes still have to match. Original manifests are not edited. See data/manifest.json and data/processed/dataset_metadata.json for the current files and direct source provenance.

## Processing commands and maintenance

Use package entry points from the repository root:

```sh
python -m data_processing.build_market_features
python -m data_processing.build_onchain_features
python -m data_processing.build_dataset --rebuild
```

The first two commands regenerate interim feature tables, so normal verification uses the retained tables instead. Acquisition is separate: `python -m data_processing.fetch_stablecoin_data` requires the optional acquisition dependencies and makes network requests. It is never invoked by the model reproduction command.

`modelling.build_corrected_dataset` remains a compatibility wrapper for the new builder. Output paths are defined in data_processing/config.py; imports do not create directories or start HTTP sessions. Writers create their own output directories. Statistical correction functions remain in modelling/corrected_data.py to preserve the existing model contract.

The root `.style.yapf` records Google-based formatting with four-space indentation and an 80-column limit. Processing modules use grouped imports and public-function docstrings. Preserve statistical definitions when formatting or moving code.

## Saved evidence verification

`python -W ignore::DeprecationWarning -m modelling.verify_release` checks the four final refit runs, input/pipeline hashes, split boundaries, validation cutoffs/budgets, independent greedy alarms, AP and yearly/pooled metrics. It writes a separate verification report, leaving frozen runs untouched. Add `--check-model-scores` to replay all final predictions and preprocessing checks; this is slower and does not retrain models. Verification scope and limitations are recorded in its JSON. Existing sensitivity runs are supporting evidence and have a separate verification status in the run index.

## Reports from saved models

`python -m reports.reproduce` combines saved h1/h6/h24 model summaries and yearly results, regenerates per-coin results, corrected events, structural types, historical-context annotations, detection counts, coverage, LR coefficients, RF importance and XGB SHAP ranking figures. It does not fit models. XGB SHAP rankings are reused from saved final-model analysis. The model comparison exporter writes to reports/model_results/ and the calibration audit to reports/validation_checks/. Optional slide support files require --presentation-dir. Robustness defaults to reports/robustness/alt_definitions/. To compare rebuilds in fresh destinations:

```sh
python -m modelling.consolidate_refit_slides --run-dir artifacts/runs/primary/REFIT-LOWEST-NEW --lr-run artifacts/runs/primary/LR-REFIT-LOWEST-NEW --reuse-shap --out-dir /private/tmp/stablecoin-presentation-rebuild --validation-dir /private/tmp/stablecoin-validation-rebuild --presentation-dir /private/tmp/stablecoin-slides-rebuild
python -m modelling.robustness_refit --run-dir artifacts/runs/primary/REFIT-LOWEST-NEW --run-dir artifacts/runs/primary/LR-REFIT-LOWEST-NEW --out-dir /private/tmp/stablecoin-robustness-rebuild
```

The presentation exporter fits benchmark models; it is separate from saved-model-only report generation. Use a fresh temporary destination for comparisons; report generation must not replace frozen input/model evidence.

## Retrain the primary protocol

Use fresh output directories. Existing runs are never overwritten.

```sh
python -m modelling.refit_after_calibration --out-dir artifacts/runs/primary/RF-XGB-REPRODUCED --models random_forest xgboost --horizon 6 --horizon 24
python -m modelling.refit_after_calibration --out-dir artifacts/runs/primary/LR-REPRODUCED --models logreg --horizon 6 --horizon 24
```

Repeat with `--horizon 1` for supplementary runs. Compare regenerated metrics to the corresponding frozen run, and verify using `--run-dir`. Practice LR chooses C from calendar-year CV within inner_train, and final refit keeps it. For the sensitivity source, `modelling.corrected_xgb` creates a fresh non-refit XGB run; `modelling.run_exhaustive_lowest --xgb-run <source> --out-dir <fresh>` recalibrates XGB and fits practice RF.

## Acquisition limitation

The six extracted Dune CSVs and their original ZIP are retained. Original market API response snapshots are unavailable; hourly_stablecoin.parquet preserves the acquired hourly table. A fresh download may differ from historical inputs. Optional acquisition dependencies are declared in pyproject.toml. Normal verification/reproduction never calls the network fetcher.

The event exporter writes grouped events and episode-to-event mappings to `reports/depeg_events/`; the coin/year heatmap goes to `reports/depeg_episodes/` (override with `--episode-out-dir`). Interpretation outputs use `log_reg/`, `random_forest/`, and `xgboost/` subfolders. The removed refit comparison and duplicate presentation RF table are not regenerated. Saved fit directories are resolved by run ID after relocation.
