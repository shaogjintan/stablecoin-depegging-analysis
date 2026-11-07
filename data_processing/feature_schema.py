"""Feature columns and tier definitions; raw levels are excluded."""

MARKET_FEATURES = [
    "peg_deviation",
    "peg_stress",  # clip(-peg_deviation, 0, 0.01): one-split "stressed" regime
    "log_return_1h", "log_return_6h", "log_return_24h",
    "realised_vol_6h", "realised_vol_24h",
    "volume_pct_change_1h", "volume_pct_change_6h", "volume_pct_change_24h",
]


ONCHAIN_FEATURES = [
    f"{metric}_logdiff_{k}h"
    for metric in ("tx_count", "tx_volume", "active_senders", "active_receivers")
    for k in (1, 6, 24)
]


TIERS = {1: MARKET_FEATURES, 2: MARKET_FEATURES + ONCHAIN_FEATURES}


META_COLS = ["coin", "hour", "episode_id"]

