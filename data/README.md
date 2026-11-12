# Data storage

| Folder | Purpose |
|---|---|
| `raw/` | Retained source inputs: the acquired hourly market table, original Dune ZIP and extracted study-coin CSVs. |
| `interim/` | Derived market and on-chain feature tables. |
| `processed/` | The final hourly modelling dataset, episode records and dataset metadata. |

`processed/master.parquet` and `processed/depeg_episodes.parquet` are the exact
inputs used by `artifacts/final_run/`, the published reviewer run.
`depeg_episodes_w_event_type.parquet` is a reporting annotation file, not an
alternative modelling input. `manifest.json` records paths, schemas and hashes;
`processed/dataset_metadata.json` records construction details and feature method.

Both `python -m data_processing.merge_and_label` and
`python -m data_processing.build_dataset` use the same direct route: retained
hourly observations plus interim feature tables -> corrected features, episodes,
labels and eligibility -> the canonical output pair. No older master is needed.
Existing outputs are validated without replacement. Use `--rebuild` to compare
a reconstruction, or `--rebuild --out-dir <fresh-directory>` to write a new pair.
New window-local volatility outputs must match exactly during comparison.

The retained source is `raw/hourly_stablecoin.parquet`; original market API
responses are unavailable. Reacquiring market data can produce different values.
The market writer and dataset builder share calendar-based feature calculations.
Volatility uses independent window calculations with centered compensated sums,
not accumulated rolling state. Missing feature observations remain unavailable.

The original on-chain bundle is `raw/dune_onchain.zip`; six extracted CSVs sit
directly under `raw/`. Their identities match their ZIP members byte-for-byte.
Missing export rows do not establish zero transfer activity. The example SQL in
`data_processing/dune_hourly_transfers.sql` aggregates one Ethereum contract,
uses an 18-decimal divisor and is not verified as the query for all six exports.
Original per-coin query IDs, export dates, contract selections and divisors
could not be confirmed; the manifest records these provenance limits.

The hourly market source, Dune ZIP and processed pair are explicitly allowed in
Git. Extracted CSVs and derived interim tables are regenerated locally.
Superseded input copies were removed. Historical input checksums remain in
`reports/validation_checks/reproducibility/reviewer_relocation.json`.
