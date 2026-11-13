# Stablecoin depeg early-warning analysis

## Overview and motivation

Stablecoins aim to maintain a one-dollar peg and are used for trading,
settlement and collateral. A downward depeg can therefore affect both holders
and connected markets. An advance warning is useful only if it gives enough
lead time to act without producing excessive false alarms.

This project studies hourly market and Ethereum transfer data for USDT, USDC,
BUSD, DAI, TUSD and PAX over the configured coverage windows within 2018-2025.
It compares logistic regression, random forest and XGBoost with simple
constant-prior and persistence benchmarks. 

## Research question

**Can market information available at hour t predict whether a stablecoin will
begin a new depeg episode within the next h hours, and does adding on-chain
information improve prediction beyond market information alone?**

We address two questions: whether market-based models provide useful
warnings relative to simple benchmarks, and whether on-chain Ethereum transfer activity
adds predictive value once market information is included. The primary horizons
are 6 and 24 hours; the 1-hour horizon is supplementary.

## Study design

- **Outcome:** a new downward depeg episode, defined by a price below $0.99
  for at least two consecutive hours, with nearby qualifying dips merged using
  the 24-hour recovery rule. The target is episode onset, not an ongoing depeg.
- **Feature comparison:** Tier 1 uses ten market features. Tier 2 adds twelve
  on-chain change features. Both tiers use the same eligible coin-hours and
  observable labels, so coverage differences do not drive the comparison.
- **Evaluation:** expanding chronological test folds cover 2021-2025, with a
  25-hour purge and 24-hour embargo. Model preprocessing and tuning use only
  the relevant training data.
- **Alarm policy:** practice models select the lowest feasible calibration
  cutoff under an average budget of one false alarm per observed coin-month.
  Final models refit on the outer training population and retain that cutoff.
- **Measures:** median lead time, episode recall, number of false alarms,
  false alarms per coin-month and average precision. Related episodes across
  coins are also grouped into events to describe shared shocks.

Interpret results within the observed sample: missing source data restricts
which episodes are targetable and event groupings do not establish causation.
See [Methodology](docs/methodology.md) for the definitions and limitations.

## Review the results

- [Final model comparisons](reports/model_results/model_comparison_summary.csv)
- [Calibration audit](reports/validation_checks/calibration_audit.csv)
- [Outcome-definition robustness](reports/robustness/alt_definitions/README.md)
- [Reproduction instructions](docs/reproduction.md)
- [Methodology and limitations](docs/methodology.md)
- [Python file inventory](docs/code_inventory.md)

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

## Running the analysis

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

Large fitted models, partition Parquets and source snapshots are supplied
separately rather than stored in Git. A clone with the retained raw inputs can
run full source reproduction; saved-model replay requires that large evidence.
Reacquiring source data can change results and is separate from reproduction.
See [detailed instructions](docs/reproduction.md).

## Repository structure

```text
data/
  raw/                 retained market observations and Dune exports
  interim/             retained market and on-chain feature tables
  processed/           modelling dataset, episodes and annotations
data_processing/       acquisition, features, labels, eligibility and splits
modelling/
  models/              estimators and preprocessing
  training/            fitting, tuning and calibration workflow
  evaluation/          scores, alarms and performance metrics
  reporting/           tables and figures
  robustness_checks/   alternative definitions and alarm rules
  validation/          saved-evidence checks
  utils/               file and checksum helpers
reports/               published tables, figures and calibration audit
artifacts/final_run/   saved models, partitions and prediction evidence
tests/                 automated code checks
docs/                  code, data, methodology and execution guides
archive/               historical material (ignored for new Git additions)
output/                fresh runs and local documents
```

Archived code is excluded from the maintained reproduction workflow. Original
run manifests and source snapshots retain execution-time paths; current input
relocations are recorded in `reports/baseline.json`. The original Dune bundle is
retained and its six study-coin CSVs can be extracted locally.

See [Modelling guide](modelling/README.md) for the pipeline structure and current command entry points.

Full source reproduction writes regenerated feature tables to
`output/reproduction/data/interim_reproduction/` (or the equivalent location
under `--out-dir`). The retained tables in `data/interim/` remain unchanged.
