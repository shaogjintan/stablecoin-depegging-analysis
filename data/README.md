# Frozen research data

`processed/master.parquet` and `processed/depeg_episodes.parquet` are the single canonical corrected modelling pair. `dataset_version=corrected_v1` records their protocol. `depeg_episodes_w_event_type.parquet` retains report annotations and is not a fallback modelling input.

Both `python -m data_processing.merge_and_label` and `python -m data_processing.build_dataset` use the same direct route: acquired hourly market snapshot + interim market/on-chain feature tables -> corrected features, episodes and labels in memory -> canonical outputs. No old master is needed or written. Existing outputs are validated without overwriting; add `--rebuild` to compare a reconstruction or `--rebuild --out-dir <fresh>` to create a fresh pair.

`interim/` holds the acquired hourly market table and the market/on-chain feature tables; `raw/dune_onchain/` holds six source CSVs. The root Dune ZIP is retained. All six CSVs match their ZIP members byte-for-byte; see `dune_export_verification` in `manifest.json`. The old original master and old episode manifest are quarantined under delete/, never used by the maintained build.

manifest.json records current paths, schemas and hashes. Original market API responses are not available. Existing corrected inputs remain byte-identical to the model evidence. Fresh rolling reductions may vary by up to 1e-9 near zero; see docs/reproduction.md for strict comparison scope. Large source snapshots/intermediates remain local artifacts and require separate distribution.

## Dune acquisition query

The user-supplied [hourly transfer query](../data_processing/dune_hourly_transfers.sql) aggregates Ethereum ERC-20 transfer events into `hour`, `tx_count`, `total_volume`, `active_senders`, and `active_receivers`, matching the on-chain importer schema. `tx_count` counts transfer events, not distinct transaction hashes. `total_volume` is token transfer volume, not USD volume. The importer renames it to `tx_volume`.

The supplied query filters one contract (`0xdC035D45d973E3EC169d2276DDab16f1e407384F`) and divides raw values by `1e18`, assuming 18 decimals. It covers 2018-01-01 inclusive through 2026-01-01 exclusive. A grouped query produces rows only for hours containing transfers; missing export hours should not automatically be interpreted as verified zero activity.

To document all six exports fully, record each export's actual contract address, decimal divisor and Dune query identifier/export date. This supplied query has not been executed here or verified as the source of every retained CSV.
