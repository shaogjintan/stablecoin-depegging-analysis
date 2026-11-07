# Reproduction

Run commands from the repository root using CPython 3.12. See the root README
for environment creation. Install `requirements.lock` (or `requirements.txt`,
which delegates to it), then run `python -m pip check`.

The lock is a pinned environment freeze, not a wheel/hash lock. The installed
macOS environment matches it; a clean installation and other operating systems
have not been validated during cleanup. `pip install .` uses unpinned project
dependencies and is not equivalent to installing the freeze.

## Complete reproduction from retained sources

Required inputs, explicitly allowed in Git:

- `data/interim/hourly_stablecoin.parquet`: retained hourly market observations.
- `data/raw/dune_onchain.zip`: original per-coin Dune exports.

Verify their hashes against `data/manifest.json`. Their inclusion in a clone
requires committing and pushing the files, not merely allowing them in Git.
The original market API responses are unavailable; the hourly table is the
retained acquisition output.

```sh
python -m reports.reproduce --full --dry-run
python -m reports.reproduce --full
# Equivalent full workflow after activating the environment:
make reproduce
```

The dry run prints the plan without writing or training. Full reproduction:

1. Copies both retained inputs into a fresh workspace.
2. Rebuilds market and on-chain features and the processed dataset.
3. Trains LR, random forest and XGBoost for h1/h6/h24 across 2021–2025.
4. Verifies the new run, including saved prediction replay.
5. Runs h6/h24 alternative-outcome robustness checks.
6. Exports comparisons, per-coin and event-type results, calibration, coverage,
   benchmarks and model interpretation. XGBoost SHAP uses the new test evidence.

Outputs share `artifacts/reproduction/`: `data/`,
`artifacts/runs/primary/ALL-MODELS/`, `reports/`, and `reproduction.json`.
The runner scopes `STABLECOIN_DATA_DIR` to child processes and passes explicit
run paths to exporters. It uses no historical fitted models or network fetcher.
Retained inputs and final evidence remain unchanged.

Full retraining is computationally intensive and has not been executed during
cleanup. Tests and a real-source preprocessing smoke run passed. The runner
rejects existing output folders, stops on a failed stage and records progress.
There is no automatic resume. Use `--out-dir <new-directory>` for another run.
Compare new reports with the retained reports using model/configuration/horizon
keys and all five headline metrics; file hashes alone do not establish changed
model performance.

## Rerun with newly downloaded market data

Reviewers may create their own hourly Parquet:

```sh
python -m pip install requests
python -m data_processing.fetch_stablecoin_data
python -m reports.reproduce --full
```

Acquisition replaces the retained hourly market table, so use a separate
checkout when preserving original inputs. External API history and coverage
can change; new downloads may produce different hashes and metrics. The market
fetcher does not acquire on-chain exports: the retained Dune ZIP is still needed.
The full runner records the actual input hashes for either route.

Acquisition forward-fills price gaps for at most three hours. Within configured
exchange coverage windows it zero-fills missing volume, including unavailable
months when other months were returned. This historical assumption cannot
distinguish quiet trading from unavailable data. Entirely empty sources are
skipped. Missing Dune observations remain missing, rather than assumed zero.

## Verify and report the retained results without retraining

These commands require the retained processed pair and final model artifacts.
Large fitted models and prediction partitions are excluded from Git; see
`artifacts/README.md`. No public model-artifact download has been published.

```sh
python -m data_processing.build_dataset
python -W ignore::DeprecationWarning -m modelling.validation.verify_model_results
python -m reports.reproduce
```

The first command validates existing processed inputs. The verifier checks
input/pipeline hashes, fit identities, split boundaries, cutoffs, alarm outcomes,
and all five yearly/pooled metrics. Add `--check-model-scores` to replay all saved
predictions. It does not retrain models or rerun LR C selection. Split checks
validate saved rows and exclusions, not the complete eligible-row population.
Threshold selection reuses the maintained engine, tested against an oracle.

Without `--full`, the report runner regenerates reports from saved evidence.
It does not fit primary models; XGBoost rankings are reused from retained
analysis. Benchmarks fit separately via `modelling.reporting.benchmarks`.
Report generation replaces generated reports, not frozen model evidence.
See `results_guide.md` for report paths and scope.

## Data building and comparison rules

`data_processing.build_market_features.calendar_market_features` is the single
calendar-based market calculation used by both the interim writer and final
builder. It preserves missing hours and masks volume changes across venue
switches. On-chain changes use shared metric/lag definitions in feature_schema;
those lag windows are separate from forecast horizons. Dune extraction checks
all recognized members before writing, reuses identical files and rejects
conflicts. Validation rejects duplicate keys and malformed observed values;
partial coin coverage is supported and reported.

Standalone feature writers replace their configured interim files:

```sh
python -m data_processing.build_market_features
python -m data_processing.build_onchain_features
python -m data_processing.build_dataset --rebuild
```

The final builder compares against existing outputs without replacing them.
Use `--rebuild --out-dir <fresh-directory>` to create a new pair and metadata.
`merge_and_label` delegates to the same builder. An incomplete output pair or
already-corrected input fails rather than silently replacing/reshifting it.

Rebuild comparison explicitly reports changes in `market_sample`,
`common_sample`, and `return_1h/6h/24h`: historical diagnostics that do not select
or supply maintained model inputs. Only rolling volatility has a numerical
tolerance, absolute 1e-9; all other columns and the episode manifest must match.
Frozen model evidence must keep its recorded input hashes. Fresh metadata records
current source paths and implementation hashes; it does not rewrite old runs.

## Training, tests and maintenance

Training entry point: `modelling.training.train_models --out-dir <new-run>`.
Defaults include all three models and horizons. Practice LR selects C by calendar
CV inside practice training; final refitting keeps C and the calibration cutoff.
Dataset preparation lives in `data_processing/dataset_transformation.py`.
`sample_preparation.corrected_folds` reads stored eligibility; the matched-sample
helper checks unique coin-hour keys and label agreement across tiers.

```sh
make test
make verify
make reports
```

Tests use synthetic examples and mutation checks; verification audits retained
research evidence. Neither substitutes for completing fresh primary retraining.
`tests/diagnose_onchain_features.py` is a historical manual diagnostic, not the
supported training sample definition. Google-style docstrings use at most 80
characters per line; `.style.yapf` records formatting preferences.

Historical cleanup records and the delete-folder dependency audit live under
`archive/maintenance/`. They describe their recorded scope, not new validation
of every later change.
