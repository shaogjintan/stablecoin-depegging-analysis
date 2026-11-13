# Data processing

Run commands from the repository root. Configuration and feature lists live in
`config.py` and `feature_schema.py`.

1. `fetch_stablecoin_data.py` acquires prices and exchange volume into
   `data/raw/hourly_stablecoin.parquet`. Fresh acquisition may change inputs.
2. `build_market_features.py` and `build_onchain_features.py` derive the two
   feature tables under `data/interim/` from retained inputs.
3. `build_dataset.py` constructs or checks `master.parquet` and
   `depeg_episodes.parquet` under `data/processed/`.

`dataset_transformation.py` defines labels, causal eligibility and bar-close
clocks. `matched_sample.py` and `sample_preparation.py` select comparable model
inputs. `splits.py` defines chronological splits and boundary exclusions.
The annotated episode Parquet is generated later by the event-report exporter.

See [execution instructions](../docs/reproduction.md).
