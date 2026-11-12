# Stablecoin depeg early-warning analysis

We test whether market information predicts depeg onsets within 1, 6 or 24 hours, and whether on-chain change features improve warnings on matched coin-hours.

The study covers USDT, USDC, BUSD, DAI, TUSD and PAX. Final results use the corrected bar-close dataset, expanding test years 2021–2025, a 25-hour purge and 24-hour embargo. Practice models select the lowest feasible validation cutoff under an average budget of one false alarm per observed coin-month. Final models refit on the outer training population and keep that cutoff. LR also keeps the C selected within practice training. The validation budget is not a test-period guarantee.

## Review the results

- [Final model comparisons](reports/model_results/model_comparison_summary.csv)
- [Calibration audit](reports/validation_checks/calibration_audit.csv)
- [Outcome-definition robustness](reports/robustness/alt_definitions/README.md)
- [Reproduction instructions](docs/reproduction.md)
- [Methodology and limitations](docs/methodology.md)
- [Python file inventory](docs/code_inventory.md)
- [Historical cleanup record](archive/maintenance/cleanup_history.md)

## Setup and checks

```sh
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.lock
python -m pip check
python -W ignore::DeprecationWarning -m unittest discover -s tests -v
```

Run these commands from the repository root. On Windows, create the environment
with `py -3.12 -m venv .venv` and activate it with
`.venv\Scripts\Activate.ps1`. The dependency freeze was checked on macOS with
CPython 3.12; installation on other platforms has not been validated.

## Reproduction and reviewer evidence

The repository retains one published run under `artifacts/final_run/`.
It contains saved models, data splits, predictions, alarms and run settings.
The exact input dataset is under `data/processed/`, and published tables and
figures are under `reports/`. `reports/baseline.json` selects these locations.

With the large saved-model evidence supplied, regenerate published reports:

```sh
python -m reports.reproduce
```

To rebuild features and data, retrain all models and create fresh reports from
the retained local sources:

```sh
python -m reports.reproduce --full
```

Fresh outputs go to `output/reproduction/`, leaving the published final run
unchanged. Existing output folders are rejected; use `--out-dir <new-folder>`
for another run. `--dry-run` previews either workflow. Full training is intensive.
The required raw inputs are `data/raw/hourly_stablecoin.parquet` and
`data/raw/dune_onchain.zip`; their hashes are recorded in `data/manifest.json`.
No fresh API calls are made by this workflow.

Two independent full reproductions passed their 90-cell model-score audits and
matched scientific results across 490 checks. Their duplicate workspaces were
removed after one validated run was retained for reviewers. Small comparison
records remain under `reports/validation_checks/reproducibility/`. Expected
run IDs and workspace paths differed; other software environments and image
bytes were not independently validated.

Some RF/XGBoost numbers changed from the earlier baseline after the numerical
fix. Superseded report and input copies were removed; the current
published CSVs use the final reproducible run. See the
[comparison and change list](docs/stable_reproduction_comparison.md).

Large fitted models, partition Parquets and source snapshots are supplied
separately rather than stored in Git. A clone with the retained raw inputs can
run full source reproduction; saved-model replay requires that large evidence.
Reacquiring source data can change results and is separate from reproduction.
See [detailed instructions](docs/reproduction.md).

## Repository

```text
modelling/             training, evaluation, robustness and reporting packages
modelling/reporting/   final-model interpretation, events and coverage
data_processing/       acquisition, feature building, sample preparation and splits
data/                  frozen inputs, source snapshots and intermediates
tests/                 maintained data/threshold/model checks
reports/               report outputs, provenance and reproduce.py
artifacts/final_run/    one final fitted run and its verification evidence
docs/                  methodology, reproduction and relocation records
output/                fresh reproduction workspaces and review deliverables
archive/       earlier experiments and original implementations
delete/                preserved deletion candidates; not used by the pipeline
```

Archived code is historical evidence and is not a supported reproduction route. Original run manifests and source snapshots retain historical paths; [the relocation map](archive/maintenance/cleanup_moves.json) maps them to current locations. `data/raw/dune_onchain.zip` and extracted Dune CSVs are both retained.

See [the modelling guide](modelling/README.md) for the pipeline structure and current command entry points.
