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
python -m data_processing.build_dataset
python -W ignore::DeprecationWarning -m unittest discover -s tests -v
python -W ignore::DeprecationWarning -m modelling.verify_release
python -m reports.reproduce
```

Verification and saved-model report generation require the local run artifacts described in [artifacts/runs/README.md](artifacts/runs/README.md). They are not all stored in Git. No network data acquisition runs as part of these commands.

## Repository

```text
modelling/             maintained final runners, shared model/data helpers
modelling/reporting/   final-model interpretation, events and coverage
data_processing/      acquisition, feature building, corrected split validation
data/                  frozen inputs, source snapshots and intermediates
tests/                 maintained data/threshold/model checks
scripts/               report reproduction entry point
reports/         authoritative report outputs and provenance
artifacts/runs/        primary, supplementary and sensitivity evidence
docs/                  methodology, reproduction and relocation records
archive/history/       earlier experiments and original implementations
delete/                preserved deletion candidates; not used by the pipeline
```

Archived code is historical evidence and is not a supported reproduction route. Original run manifests and source snapshots retain historical paths; [the relocation map](docs/cleanup_moves.json) maps them to current locations. `dune_onchain.zip` and extracted Dune CSVs are both retained.
