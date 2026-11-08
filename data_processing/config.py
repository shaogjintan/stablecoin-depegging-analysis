"""Configuration for the stablecoin depeg-forecasting data pipeline."""

from pathlib import Path
import os

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
# The full reproduction command scopes this root to its output workspace.
# Without an override, existing commands continue to use retained repo inputs.
DATA_DIR = Path(os.environ.get("STABLECOIN_DATA_DIR", "data"))
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
HOURLY_MARKET_PATH = INTERIM_DATA_DIR / "hourly_stablecoin.parquet"
MARKET_FEATURES_PATH = (
    INTERIM_DATA_DIR / "hourly_stablecoin_market_features.parquet"
)
ONCHAIN_FEATURES_PATH = (
    INTERIM_DATA_DIR / "hourly_stablecoin_onchain_features.parquet"
)
DUNE_ONCHAIN_ZIP = RAW_DATA_DIR / "dune_onchain.zip"
DUNE_ONCHAIN_DIR = RAW_DATA_DIR
PROCESSED_DATA_DIR = DATA_DIR / "processed"
# Canonical corrected inputs; dataset_version records the statistical protocol.
MODELLING_DATA_VERSION = "corrected_v1"
ACTIVE_MASTER_PATH = PROCESSED_DATA_DIR / "master.parquet"
ACTIVE_EPISODES_PATH = PROCESSED_DATA_DIR / "depeg_episodes.parquet"
ACTIVE_TYPED_EPISODES_PATH = (
    PROCESSED_DATA_DIR / "depeg_episodes_w_event_type.parquet"
)
DATASET_METADATA_PATH = PROCESSED_DATA_DIR / "dataset_metadata.json"

# ---------------------------------------------------------------------------
# Universe
# ---------------------------------------------------------------------------

# Study stablecoins; DAI is crypto-collateralised, not fiat-backed.
FIAT_COINS = ["USDT", "USDC", "BUSD", "DAI", "TUSD", "PAX"]

# Per-coin windows (UTC). No "end" -> END_DATE. Start = first day DefiLlama has
# genuinely hourly prices (>= 90% of hours observed for 30 days) and, where
# later, the first day of Dune on-chain data. Before 2018-06-27 CoinGecko only
# has one price every ~6 hours.
DEFILLAMA_FIAT_COIN_SPECS = {
    # hourly from 2018-06-27, but Jul-Aug 2018 prices are noise (99th pct
    # hourly move ~10%, single-hour swings $0.75-$1.25 vs <1% from Sep 2018).
    # Starts after that; still includes the Oct 2018 depeg (low $0.944)
    "USDT": {"id": "coingecko:tether", "start": "2018-09-01"},
    # first price 2018-10-04 16:00 (launched late Sep 2018), hourly from
    # 2018-10-05
    "USDC": {"id": "coingecko:usd-coin", "start": "2018-10-05"},
    # multi-collateral DAI launch; trades ~$0.97-0.98 until mid-Jan 2020
    "DAI": {"id": "coingecko:dai", "start": "2019-11-18"},
    # hourly from 2018-09-26; USDP still trades on-peg with on-chain activity
    # through 2025, so no early end
    "PAX": {"id": "coingecko:paxos-standard", "start": "2018-09-26"},
    # prices hourly from 2018-06-27, but Dune on-chain data (current TUSD
    # contract) only begins 2019-01-04
    "TUSD": {"id": "coingecko:true-usd", "start": "2019-01-04"},
    # hourly from 2019-09-21; minting halted Feb 2023, Paxos support through
    # Feb 2024, on-chain transfers fall ~80% in Mar 2024 and price turns
    # erratic ($0.73-$2.40, 20-47% of hours < $0.99) from Jun 2024
    "BUSD": {
        "id": "coingecko:binance-usd",
        "start": "2019-09-21",
        "end": "2024-02-29 23:00",
    },
}

# Exchange OHLCV supplies volume and venue provenance. Reference prices,
# price-based features and labels use the DefiLlama series above.
# USDT and DAI use USD quotes on Bitfinex to avoid self/stablecoin quotes;
# the other four use USDT quotes on Binance. USDT-quoted volume is measured
# in that trading pair's base units, not independent USD notional.
EXCHANGE_OHLCV_SPECS = {
    "USDT": [{"venue": "bitfinex", "symbol": "tUSTUSD", "quote": "USD",
              "start": "2018-11-19"}],
    "USDC": [{"venue": "binance", "symbol": "USDCUSDT", "quote": "USDT",
              "start": "2018-12-01"}],
    "DAI": [{"venue": "bitfinex", "symbol": "tDAIUSD", "quote": "USD",
             "start": "2019-11-18"}],
    # Paxos Standard renamed PAX -> USDP; Binance swapped the pair 2021-09
    "PAX": [{"venue": "binance", "symbol": "PAXUSDT", "quote": "USDT",
              "start": "2018-12-01", "end": "2021-09-30 23:00"},
            {"venue": "binance", "symbol": "USDPUSDT", "quote": "USDT",
              "start": "2021-10-01"}],
    "TUSD": [{"venue": "binance", "symbol": "TUSDUSDT", "quote": "USDT",
              "start": "2018-06-01"}],
    "BUSD": [{"venue": "binance", "symbol": "BUSDUSDT", "quote": "USDT",
              "start": "2019-09-01", "end": "2023-12-31 23:00"}],
}

# ---------------------------------------------------------------------------
# Study window
# ---------------------------------------------------------------------------
START_DATE = "2018-01-01"
END_DATE = "2025-12-31 23:00"  # inclusive

# ---------------------------------------------------------------------------
# Depeg definition
# ---------------------------------------------------------------------------
DEPEG_BAND = 0.01
DEPEG_MIN_DURATION_HOURS = 2
# Merge qualifying dips within 24h of recovery onset; also exclude origins
# through that recovery window in the causal risk mask.
DEPEG_GAP_HOURS = 24
FORECAST_HORIZONS_HOURS = [1, 6, 24]

# ---------------------------------------------------------------------------
# Leakage control
# ---------------------------------------------------------------------------
# Drop the first 24h after each split boundary for feature lookback.
EMBARGO_HOURS = 24
PURGE_HOURS = max(FORECAST_HORIZONS_HOURS) + DEPEG_MIN_DURATION_HOURS - 1
# Calendar test years (UTC). splits.py applies purge, embargo, and
# boundary-episode exclusions; calibration uses the preceding year.
OOS_START_YEAR = 2021
OOS_END_YEAR = 2025
