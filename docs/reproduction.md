# Reproduction

Run commands from the repository root using CPython 3.12. `requirements.lock` records all packages in the local macOS environment used for cleanup validation; it is an environment freeze, not a wheel/hash lock. The original Windows requirements are preserved under archive/history/documentation/. Cross-platform numerical differences should be assessed with tolerances rather than assuming byte-identical model files.

## Fresh environment and required files

Create and activate a CPython 3.12 virtual environment as shown in the root
README, install `requirements.lock`, and run `python -m pip check`. Running
modules from the repository root does not require an editable installation.
`requirements.txt` delegates to the same freeze; use either requirements file,
not two separate dependency installations. `pip install .` uses the unpinned
project dependencies and is not equivalent to recreating the recorded freeze.
The `[acquisition]` extra is needed only for optional network downloads.

The local installed versions match the freeze and pass `pip check`. A clean
installation from a package index and other operating systems have not been
tested during this cleanup; no fresh-environment installation is claimed.

The Dune ZIP and frozen processed pair are tracked. The Git rules now also
allow `data/interim/hourly_stablecoin.parquet`, the retained 7.9 MiB market
source needed by `--full`. Include this file in the repository commit before
distributing a clone; an ignore-rule change does not stage or commit it.
If a distributed copy omits it, supply the source bundle or use fresh
acquisition. Verify the retained source checksum against `data/manifest.json`.
Saved-model verification additionally needs the omitted run evidence described
in `artifacts/README.md`. There is no published artifact download yet.
Reacquiring market data can yield different observations and does not replace
the missing original source for an exact-input reproduction.

## Rerun with newly downloaded market data

Reviewers can create the hourly Parquet themselves. After the pinned environment
setup, install the optional HTTP dependency and run acquisition from the repo
root:

```sh
python -m pip install requests
python -m data_processing.fetch_stablecoin_data
python -m reports.reproduce --full
```

The acquisition command writes `data/interim/hourly_stablecoin.parquet` from
DefiLlama prices and exchange volume. It replaces an existing hourly file, so
use a separate checkout for a fresh-data rerun if retaining the original input.
The Dune ZIP is still required; the market fetcher does not download on-chain
exports. Network access and source availability are required for acquisition.

This route reproduces the analysis procedure using the downloaded observations.
The original source bundle instead supplies the exact retained hourly input.
External history, API responses and coverage can change; a fresh acquisition is
not guaranteed to reproduce the original checksum or metrics. The full runner
records the actual source hashes in its output workspace in either case.

## Complete local-source reproduction

```sh
python -m reports.reproduce --full
# Equivalent after activating the environment:
make reproduce
```

The command requires `data/interim/hourly_stablecoin.parquet` and the retained
root `dune_onchain.zip`. It copies the hourly observations into a fresh
workspace, extracts on-chain CSVs there, rebuilds both feature tables and the
processed dataset, then trains LR/RF/XGBoost for all three horizons and five
test years. It verifies the new run including model-score replay, runs primary
h6/h24 outcome-definition robustness checks, and exports summaries, per-coin
and event-type results, calibration, coverage, benchmarks and interpretation.
XGBoost SHAP is computed from the newly fitted models and saved test features.
No historical fitted models or saved SHAP rankings are used in this mode.

Outputs share one root, `artifacts/reproduction/`: `data/`,
`artifacts/runs/primary/ALL-MODELS/`, and `reports/`. The workflow sets
`STABLECOIN_DATA_DIR` only for child processes so builders, training and reports
read the same fresh dataset. Explicit `--run-dir` arguments select the new run;
the per-coin exporter searches only the fresh run root. The parent shell and
retained source/frozen result paths are unchanged.

Use `--full --dry-run` to inspect stages without writing or training. Use
`--full --out-dir <new-directory>` for a different destination. Existing output
folders are rejected; failed runs remain available for diagnosis. The workflow
stops on the first failure and records completed stages, commands and source
hashes in `reproduction.json`. A complete status requires every stage to pass.
There is no automatic resume or network acquisition. Full model retraining
can take substantial time; the dry run is not evidence of completed training.

Compare the new `reports/model_results/model_comparison_summary.csv` with the
retained report using model, configuration and horizon keys and the five
headline metrics. Input/output hashes and environment metadata explain their
provenance; changed file hashes alone do not establish changed model results.

Without `--full`, `python -m reports.reproduce` retains the saved-evidence
report-only workflow described below.

## Frozen input validation

Both entry points use the same direct build path:

```sh
python -m data_processing.build_dataset
python -m data_processing.merge_and_label
```

They validate the existing canonical `data/processed/master.parquet` and `depeg_episodes.parquet`. If missing, they merge the acquired hourly snapshot and two interim feature tables, then apply corrected features/labels entirely in memory and write only the corrected canonical pair plus metadata. No original master or legacy episode file is needed or produced.

Add `--rebuild` to rebuild and compare with existing files without replacing them; use `--rebuild --out-dir <fresh-directory>` to write a new pair. Incomplete pairs fail explicitly. Already-corrected input is rejected to prevent shifting clocks twice. The build preserves labels, episode definitions and model sample populations. The comparison explicitly reports differences in the two historical completeness flags (`market_sample`, `common_sample`) and three simple-return diagnostics (`return_1h`, `return_6h`, `return_24h`). These columns do not select the maintained model sample or supply model features. Source-block layout can change rolling standard deviations by up to 1e-9 in absolute value near zero; those two columns use that tolerance. Every remaining column must match exactly. Existing frozen canonical bytes and their model results remain unchanged. A freshly rebuilt file may have a different checksum and should not be silently substituted for frozen model evidence.

Historical manifest filenames resolve to the canonical pair through an explicit alias map; hashes still have to match. Original manifests are not edited. See data/manifest.json and data/processed/dataset_metadata.json for the current files and direct source provenance.

## Processing commands and maintenance

Use package entry points from the repository root:

```sh
python -m data_processing.build_market_features
python -m data_processing.build_onchain_features
python -m data_processing.build_dataset --rebuild
```

These are rebuild commands: the first two replace interim feature tables using retained local sources, and the third compares a final reconstruction without replacing existing processed files. The market writer uses the current calendar calculation, so its output can differ from the retained historical interim table. This is a reproducible current calculation, not a promise to recreate historical bytes. For frozen evidence verification, skip the interim writers and use the saved-evidence verifier below; it checks recorded hashes. For fresh processed outputs, use a new `--out-dir` and retain their new metadata separately.

Acquisition is separate: `python -m data_processing.fetch_stablecoin_data` requires the optional acquisition dependencies and makes network requests. It is never invoked by the model reproduction command.

The obsolete `modelling.build_corrected_dataset` wrapper is archived; use `data_processing.build_dataset`. Output paths are defined in data_processing/config.py; imports do not create directories or start HTTP sessions. Writers create their own output directories. Dataset correction functions live in data_processing/dataset_transformation.py; feature definitions, sample preparation, and matched populations also live in data_processing/.

The root `.style.yapf` records Google-based formatting with four-space indentation and an 80-column limit. Processing modules use grouped imports and public-function docstrings. Preserve statistical definitions when formatting or moving code.

## Saved evidence verification

`python -W ignore::DeprecationWarning -m modelling.validation.verify_model_results` checks the four final refit runs, input/pipeline hashes, split boundaries, validation cutoffs/budgets, independent greedy alarms, all five yearly/pooled headline metrics. It writes a separate verification report, leaving frozen runs untouched. Add `--check-model-scores` to replay all final predictions and preprocessing checks; this is slower and does not retrain models. Verification scope and limitations are recorded in its JSON. Existing sensitivity runs are supporting evidence and have a separate verification status in the run index.

## Reports from saved models

`python -m reports.reproduce` combines saved h1/h6/h24 model summaries and yearly results, regenerates per-coin results, corrected events, structural types, historical-context annotations, detection counts, coverage, LR coefficients, RF importance and XGB SHAP ranking figures. It does not fit models. XGB SHAP rankings are reused from saved final-model analysis. The model comparison exporter writes to reports/model_results/ and the calibration audit to reports/validation_checks/. Robustness defaults to reports/robustness/alt_definitions/. To compare rebuilds in fresh destinations:

```sh
python -m modelling.robustness_checks.alternative_definitions --run-dir artifacts/h6_and_h24/random_forest_xgboost --run-dir artifacts/h6_and_h24/logistic_regression --out-dir /private/tmp/stablecoin-robustness-rebuild
```

The separate `modelling.reporting.benchmarks` exporter fits benchmark models. Optional presentation exporters are archived under archive/history/modelling/reporting/. Use a fresh temporary destination for comparisons; report generation must not replace frozen input/model evidence.

## Retrain the primary protocol

Use fresh output directories. Existing runs are never overwritten.

```sh
python -m modelling.training.train_models --out-dir artifacts/runs/primary/RF-XGB-REPRODUCED --models random_forest xgboost --horizon 6 --horizon 24
python -m modelling.training.train_models --out-dir artifacts/runs/primary/LR-REPRODUCED --models logreg --horizon 6 --horizon 24
```

Repeat with `--horizon 1` for supplementary runs. Compare regenerated metrics to the corresponding frozen run, and verify using `--run-dir`. Practice LR chooses C from calendar-year CV within inner_train, and final refit keeps it. The historical RF/XGBoost no-refit runners are archived under `archive/history/modelling/sensitivity/`; their saved run evidence is retained. They are excluded from the maintained reproduction workflow.

## Acquisition limitation

The six extracted Dune CSVs and their original ZIP are retained. Original market API response snapshots are unavailable; hourly_stablecoin.parquet preserves the acquired hourly table. A fresh download may differ from historical inputs. Optional acquisition dependencies are declared in pyproject.toml. Normal verification/reproduction never calls the network fetcher.

The on-chain builder reuses byte-identical extracted CSVs. It checks all
recognized ZIP members before extraction and rejects duplicate coin exports,
unsafe paths, and conflicts with existing files. CSV validation rejects missing
columns, duplicate coin-hour rows, invalid timestamps, and negative or infinite
observations. Missing metric observations remain missing. Partial coin coverage
is supported and reported; it does not imply complete coverage of all six coins.

The event exporter writes grouped events and episode-to-event mappings to `reports/depeg_events/`; the coin/year heatmap goes to `reports/depeg_episodes/` (override with `--episode-out-dir`). Interpretation outputs use `log_reg/`, `random_forest/`, and `xgboost/` subfolders. The removed refit comparison and duplicate presentation RF table are not regenerated. Saved fit directories are resolved by run ID after relocation.

Generate the calibration audit independently, without fitting models: `python -m modelling.reporting.calibration_audit`. It retains the existing primary h6/h24 scope; add `--include-h1` for supplementary calibration evidence. The saved-model reproduction command includes the primary audit. Benchmarks have a separate entry point, `python -m modelling.reporting.benchmarks`, which fits benchmark models only.

## Tests versus saved-run validation

`make test` checks implementation behaviour with small examples, including negative tests that deliberately alter each headline metric, duplicate a summary cell, or mismatch fit metadata. These tests also check that pooled rates use summed denominators and that pooled lead time uses individual alarms rather than yearly medians.

`make verify` audits the retained research artifacts: input/pipeline hashes, fit identities, unique yearly/pooled result cells, calibration cutoffs and budgets, independently reconstructed alarms, and all five headline metrics. It requires the saved run files and can take substantially longer. Add `--check-model-scores` to the verifier to replay model predictions as well. Verification does not retrain models or rerun LR C selection. Split checks now cover all four saved partitions: practice training, calibration, final training, and test. They enforce the 25h purge, 24h embargo, both calendar boundaries, episode exclusions, duplicate-origin rejection, and practice-partition membership in final training. They validate saved rows; they do not reconstruct the entire eligible-row population.

The verifier is production audit code under `modelling/validation/`; tests of that verifier live in `tests/test_saved_run_verification.py`. Shared checksum/path utilities remain under `modelling/utils/` because both training and reporting use them.

Validation modules use descriptive names: `verify_model_results.py` is the audit command, and `reference_checks.py` supplies independent calculations of metrics, alarms, and preprocessing expectations. An episode is targetable when at least one eligible prediction origin falls before onset and within the forecast horizon; targetable episodes form the episode-recall denominator.

## Canonical sample preparation

`data_processing.sample_preparation.corrected_folds` is the single maintained fold builder. `load_folds` loads the canonical pair and delegates to it. Both require nonmissing boolean eligibility flags for the requested tier/horizon; older generic flags and retrospective risk-mask fallbacks are not accepted. Features are read from the canonical master rather than rebuilt during sample preparation.

The manual `tests.diagnose_onchain_features` script retains its historical retrospective sample mask locally for diagnostic comparison continuity. That mask is not used by training or outcome-definition robustness checks. The original sample-preparation module is preserved under `archive/history/data_processing/`.

## Shared market-feature calculations

`data_processing.build_market_features.calendar_market_features` owns the canonical calendar-based market calculation. `dataset_transformation.prepare_corrected` calls that function before constructing onset labels, causal eligibility, and completed-bar timestamps. Fresh dataset metadata records hashes for both source modules.

The market writer and final dataset builder now use the same calendar-based
calculation. Historical price/exchange builders and the unused old eligibility
helper are archived under `archive/history/data_processing/`. Fresh builds
recalculate `market_sample` and `common_sample` from current features; training
continues to use `prediction_eligible_*`. Coverage reports also derive current
completeness directly, including when reading a frozen master.

Frozen datasets and run hashes remain unchanged. `--rebuild` explicitly reports
changes in the historical completeness flags and simple-return diagnostics.
All other fields must still match, apart from the documented rolling-volatility
tolerance. These diagnostics are not model features. Fresh output pairs have
new hashes and must not replace the recorded inputs of frozen model runs.

The maintained label implementation is
`data_processing.dataset_transformation.episode_labels`: it labels new merged
episode onsets, not every qualifying dip within an existing episode. The
unused older `merge_and_label.build_labels` implementation is archived in
`archive/history/data_processing/merge_and_label_before_label_cleanup.py`.
Merge/grid helpers, episode construction, and completeness flags remain active.

Episode thresholds, minimum dip duration, recovery/merge windows, and forecast
horizons are defined in `data_processing/config.py`. The transformation module
uses those defaults for labels, causal risk masks, and eligibility columns;
explicit helper overrides remain available for tests. On-chain feature lag
windows are separate feature settings, not forecast horizons.

## Matched feature-tier comparisons

`data_processing.matched_sample._common_sample_fold` restricts each market-only
partition to the coin-hour keys eligible for the market plus on-chain tier.
It matches train, inner training, calibration, and test separately, preserving
market row order and features. It requires the same test year and horizon,
unique nonmissing keys, aligned row counts, on-chain keys contained in the
market population, and identical labels at matched keys. This controls sample
composition when comparing feature tiers; it does not use the historical
`common_sample` completeness column or match on outcome values.

## Acquisition assumptions

The optional fetcher forward-fills reference-price gaps for at most three
hours. Within each configured exchange coverage window, it zero-fills missing
volume, including unavailable archive months if other months were returned.
That historical rule does not establish whether a missing candle reflects
zero activity or unavailable data. Entirely empty exchange sources are skipped.
A new acquisition replaces the hourly output and can differ from the retained
research input. No downloads are required for frozen-evidence verification.
