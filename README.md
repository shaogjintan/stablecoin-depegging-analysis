# Stablecoin depeg early-warning analysis

NUS DSE4101, Group 1. We test whether market information predicts depeg onsets within 1, 6 or 24 hours, and whether on-chain change features improve warnings on matched coin-hours.

The study covers USDT, USDC, BUSD, DAI, TUSD and PAX. Final results use the corrected bar-close dataset, expanding test years 2021–2025, a 25-hour purge and 24-hour embargo. Practice models select the lowest feasible validation cutoff under an average budget of one false alarm per observed coin-month. Final models refit on the outer training population and keep that cutoff. LR also keeps the C selected within practice training. The validation budget is not a test-period guarantee.

## Review the results

- [Final model comparisons](reports/model_results/model_comparison_summary.csv)
- [Calibration audit](reports/validation_checks/calibration_audit.csv)
- [Outcome-definition robustness](reports/robustness/alt_definitions/README.md)
- [Reproduction instructions](docs/reproduction.md)
- [Methodology and limitations](docs/methodology.md)
- [Cleanup record and outstanding work](docs/cleanup_checklist.md)

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

A Git clone alone does **not** contain every required research artifact:

| Workflow | Additional files needed |
|---|---|
| Full reproduction | `data/interim/hourly_stablecoin.parquet`, supplied separately; `dune_onchain.zip` is tracked in Git |
| Verify saved models or regenerate their reports | Retained inputs and saved run evidence described in [artifacts/runs/README.md](artifacts/runs/README.md) |

There is currently no published download for the separately supplied files.
Check their paths and SHA-256 hashes against [data/manifest.json](data/manifest.json).
Installing Python packages does not supply research data or saved models.

For local handoff, `artifacts/distribution/source_data.zip` contains the hourly
source, `SOURCE_DATA_SHA256SUMS`, and `SOURCE_DATA_README.md`. Supply the ZIP and
its companion `source_data.zip.sha256` alongside this repository. Extract into
the repository root without overwriting existing files, then follow the bundled
checksum and reproduction instructions. This bundle is generated locally and
is not included in Git; a public download link has not been published.

## Reproduce from retained source data

After setup, run the complete workflow with:

```sh
python -m reports.reproduce --full
```

This rebuilds features and the dataset, trains logistic regression, random
forest and XGBoost for h1/h6/h24 across 2021–2025, verifies the new predictions,
and generates robustness checks, benchmarks and reports. It requires the
retained `data/interim/hourly_stablecoin.parquet` and `dune_onchain.zip`, not
historical fitted models. It does not download updated data.

All fresh outputs go under `artifacts/reproduction/`, leaving the retained
research evidence unchanged. Preview stages with `--full --dry-run`. An existing
output folder is rejected; use `--out-dir <new-folder>` for another run. Full
retraining is computationally intensive. See the [workflow details](docs/reproduction.md).

With the separately supplied saved-model artifacts in place, verify or regenerate
the original reports without retraining:

```sh
python -m data_processing.build_dataset
python -W ignore::DeprecationWarning -m modelling.validation.verify_model_results
python -m reports.reproduce
```

## Repository

```text
modelling/             training, evaluation, robustness and reporting packages
modelling/reporting/   final-model interpretation, events and coverage
data_processing/       acquisition, feature building, sample preparation and splits
data/                  frozen inputs, source snapshots and intermediates
tests/                 maintained data/threshold/model checks
reports/               report outputs, provenance and reproduce.py
artifacts/runs/        primary, supplementary and sensitivity evidence
docs/                  methodology, reproduction and relocation records
archive/history/       earlier experiments and original implementations
delete/                preserved deletion candidates; not used by the pipeline
```

Archived code is historical evidence and is not a supported reproduction route. Original run manifests and source snapshots retain historical paths; [the relocation map](docs/cleanup_moves.json) maps them to current locations. `dune_onchain.zip` and extracted Dune CSVs are both retained.

See [the modelling guide](modelling/README.md) for the pipeline structure and current command entry points.
