# Reproduction

Run package commands from the repository root with CPython 3.12. The dependency
freeze was checked in the local macOS environment; other platforms and software
versions have not been validated. A lock file fixes versions, not wheel hashes.

```sh
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.lock
python -m pip check
python -W ignore::DeprecationWarning -m unittest discover -s tests -v
```

## File layout

- `data/raw/`: retained market observations and original on-chain exports.
- `data/interim/`: derived feature tables.
- `data/processed/`: the exact dataset and episodes used by the final run.
- `artifacts/final_run/`: one saved run, including models and prediction evidence.
- `reports/`: published tables, figures and verification records.
- `reports/baseline.json`: selected final run, data root and input relocations.

## Regenerate the published reports from saved models

With the final run evidence supplied, run:

```sh
python -m reports.reproduce --dry-run
python -m reports.reproduce
```

The command verifies the selected run identity and uses `data/` as the data root.
It verifies saved model scores, evaluates robustness checks and generates the
reports without refitting the primary models. Simple benchmarks are evaluated
separately on saved partitions. Default exporters select the published run.

Large model files, prediction Parquets and the original source snapshot are
ignored by Git. A clone with only small result files can inspect the published
CSV results but needs the supplied large evidence for model-score replay or
full saved-report regeneration. There is no public model download configured.

To verify the saved evidence directly:

```sh
python -m modelling.validation.verify_model_results --check-model-scores
```

Original `run.json` and model evidence are preserved byte-for-byte. Historical
input paths are mapped explicitly to the canonical dataset by
`reports/baseline.json` and `artifacts/final_run/input_paths.json`. Verification
checks the resolved files against the original training hashes; an arbitrary
missing path is not silently mapped by its filename.

## Rebuild data, retrain and generate all results

Required retained source inputs:

- `data/raw/hourly_stablecoin.parquet`
- `data/raw/dune_onchain.zip`

Verify their hashes against `data/manifest.json`. Both files are required in the supplied review materials.
Installing dependencies does not supply research data.

```sh
python -m reports.reproduce --full --dry-run
python -m reports.reproduce --full
```

The default fresh workspace is `output/reproduction/`, containing rebuilt data,
new saved runs, verification and generated reports. It uses retained local
inputs, not fresh API requests or the supplied fitted models. Existing output
folders are rejected. To run another independent reproduction:

```sh
python -m reports.reproduce --full --out-dir output/reproduction-check
```

The workflow builds market/on-chain features and datasets, trains all three
models for h1/h6/h24 and 2021-2025, audits all 90 cells with model-score replay,
and creates alternative-definition and alarm-rule robustness outputs, model
comparisons, calibration audits, event tables, coverage, benchmarks and
interpretation. The event exporter also regenerates
`data/processed/depeg_episodes_w_event_type.parquet` within that workspace,
using the same classifications as the event report tables. Primary training is
computationally intensive. `reproduction.json`
records source hashes, commands, completed stages and failure/completion status.
The selected final run and published report outputs are not replaced.

To isolate training from source reconstruction, optionally run:

```sh
python -m reports.reproduce --full --from-processed \
  --out-dir output/reproduction-from-processed
```

This copies the canonical processed pair and metadata, then runs fresh training
and reporting. It is not a raw-source rebuild and is labelled accordingly in its
workflow record. It uses the same published processed inputs, not older data.

## Dataset-only rebuilding

```sh
python -m data_processing.build_dataset
python -m data_processing.build_dataset --rebuild
```

The first command validates the canonical pair; the second reconstructs and
compares without replacing it. Use `--rebuild --out-dir <fresh-directory>` to
write a new pair. The market and on-chain interim tables can be regenerated
from retained source inputs. An incomplete pair or already shifted input fails
rather than being silently repaired or shifted twice.

New metadata records the window-local volatility method. Those datasets require
exact feature equality on rebuild. The legacy comparison mode retains a 1e-9
volatility allowance only for historical datasets without that method marker.
No current published input relies on that allowance.

## Fresh market acquisition and on-chain provenance

`python -m data_processing.fetch_stablecoin_data` uses external APIs and
replaces the retained hourly market source. Acquisition requires optional
network dependencies and source availability. Newly downloaded historical
values and coverage may differ from the supplied observations, so this reruns
the method rather than reproducing the same inputs. Do not overwrite retained
sources when checking reported results.

The original per-coin Dune query settings could not be fully confirmed. The SQL
file is an example, not verified six-coin acquisition code. Query access, credits
and export limits can constrain reacquisition. Retained ZIP/CSV inputs support
same-input reproduction without rerunning Dune queries. Missing exports cannot
be interpreted as verified zero transfer activity.

## Organisation and maintenance

`modelling.training.train_models --out-dir <new-run>` is the direct training
entry point. Fixed tree settings live in `training/parameters.py`; LR tuning
uses calendar validation within practice training. Dataset preparation lives
in `data_processing/dataset_transformation.py`; matched-sample checks enforce
common origins and labels across tiers. Tests check implementation behavior;
verification audits saved research evidence. Historical code is not a supported
runtime dependency. See `docs/code_organisation.md` for report wording.
