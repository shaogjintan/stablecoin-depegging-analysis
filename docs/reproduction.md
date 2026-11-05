# Reproduction

Run commands from the repository root using CPython 3.12. `requirements.lock` records all packages in the local macOS environment used for cleanup validation; it is an environment freeze, not a wheel/hash lock. The original Windows requirements are preserved under archive/history/documentation/. Cross-platform numerical differences should be assessed with tolerances rather than assuming byte-identical model files.

## Frozen input validation

`python -m modelling.build_corrected_dataset` validates the existing corrected inputs. If absent, the builder uses data/snapshots/master_original.parquet. It must not apply bar-close correction twice. See data/manifest.json for checksums and docs/cleanup_moves.json for relocations. Frozen metadata retains its original source paths.

## Saved evidence verification

`python -W ignore::DeprecationWarning -m modelling.verify_release` checks the four final refit runs, input/pipeline hashes, split boundaries, validation cutoffs/budgets, independent greedy alarms, AP and yearly/pooled metrics. It writes a separate verification report, leaving frozen runs untouched. Add `--check-model-scores` to replay all final predictions and preprocessing checks; this is slower and does not retrain models. Verification scope and limitations are recorded in its JSON. Existing sensitivity runs are supporting evidence and have a separate verification status in the run index.

## Reports from saved models

`python -m scripts.reproduce` regenerates corrected events, structural types, historical-context annotations, detection counts, coverage, LR coefficients, RF importance and XGB SHAP ranking figures. It does not fit models. XGB SHAP rankings are reused from saved final-model analysis. Existing authoritative presentation/robustness tables are preserved; to rebuild them use fresh destinations:

```sh
python -m modelling.consolidate_refit_slides --run-dir artifacts/runs/primary/REFIT-LOWEST-NEW --lr-run artifacts/runs/primary/LR-REFIT-LOWEST-NEW --reuse-shap --out-dir /private/tmp/stablecoin-presentation-rebuild
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
