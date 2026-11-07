# Data contract

Exact columns, row counts and file checksums are recorded in data/manifest.json. The maintained feature lists are data_processing.feature_schema.MARKET_FEATURES and ONCHAIN_FEATURES; they are the executable source of truth.

- coin: study stablecoin identifier, including DAI and the historical PAX naming.
- hour: corrected completed-bar UTC timestamp.
- bar_start: original source bar timestamp, one hour before the corrected hour.
- price: source price used for depeg definitions.
- trading_volume: retained exchange volume; missing values remain missing.
- episode_id, start, end: merged episode identity and corrected boundaries.
- label_h1, label_h6, label_h24: future merged onset targets. Eligibility also depends on label observability.
- prediction_eligible_*: versioned at-risk/sample eligibility masks by tier and horizon.
- observable_at_risk: causal online eligibility, distinct from future-label observability.
- *_logdiff_{1h,6h,24h}: on-chain log1p changes, used instead of raw activity levels as model features.
- historical_context (report output): heuristic date-based context, not a causal label.
- type (report output): structural event category derived from grouped episode geometry and severity/duration rules.

Raw on-chain levels are used for source-coverage descriptions, not model features. Coverage outputs explicitly distinguish observed volume, raw on-chain field completeness, and completeness of current engineered features. None includes the entire causal-risk/label eligibility rule; use the eligibility masks and saved run partitions for model populations.
