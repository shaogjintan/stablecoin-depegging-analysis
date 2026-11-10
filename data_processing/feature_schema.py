"""Define feature tiers and on-chain inputs; tiers exclude raw levels.

The on-chain builder, dataset column order, and coverage report share these
definitions. Change lags are calendar hours, using a complete hourly grid.
"""

MARKET_FEATURES = [
    "peg_deviation",
    "peg_stress",  # clip(-peg_deviation, 0, 0.01): one-split "stressed" regime
    "log_return_1h", "log_return_6h", "log_return_24h",
    "realised_vol_6h", "realised_vol_24h",
    "volume_pct_change_1h", "volume_pct_change_6h", "volume_pct_change_24h",
]


# Keep this order stable for dataset columns and saved model inputs.
ONCHAIN_RAW_METRICS = [
    "tx_count", "tx_volume", "active_senders", "active_receivers"
]
ONCHAIN_CHANGE_LAGS = (1, 6, 24)
ONCHAIN_FEATURES = [
    f"{metric}_logdiff_{lag}h"
    for metric in ONCHAIN_RAW_METRICS
    for lag in ONCHAIN_CHANGE_LAGS
]


TIERS = {1: MARKET_FEATURES, 2: MARKET_FEATURES + ONCHAIN_FEATURES}


META_COLS = ["coin", "hour", "episode_id"]

