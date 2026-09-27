"""
Configuration for the stablecoin depeg-forecasting data pipeline.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
RAW_DATA_DIR = Path("data/raw")
SNAP_TXN_DIR = RAW_DATA_DIR / "snap_erc20" / "transfers"
SNAP_PRICE_DIR = RAW_DATA_DIR / "snap_erc20" / "prices"
SNAP_EVENT_FILE = RAW_DATA_DIR / "snap_erc20" / "event_data.csv"
DUNE_ONCHAIN_ZIP = Path("dune_onchain.zip")  # per-coin Dune exports (repo root)
DUNE_ONCHAIN_DIR = RAW_DATA_DIR / "dune_onchain"
PROCESSED_DATA_DIR = Path("data/processed")
REPORT_DIR = Path("reports")

for d in [PROCESSED_DATA_DIR, REPORT_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Universe
# ---------------------------------------------------------------------------

# Fiat-backed coins
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
    # first price 2018-10-04 16:00 (launched late Sep 2018), hourly from 2018-10-05
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
    "BUSD": {"id": "coingecko:binance-usd", "start": "2019-09-21", "end": "2024-02-29 23:00"},
}

# Single-exchange hourly OHLCV, used only for market features (returns, high-low
# spread, volume, taker-buy share) -- NOT for price/labels, which stay on the
# DefiLlama series above. USDT and DAI have no liquid USDT-quoted pair (they'd be
# pricing the coin against itself / against another stablecoin with its own
# wobble), so they're quoted in USD on Bitfinex; the other four are quoted in
# USDT on Binance, which has far deeper liquidity than any direct USD pair for
# them. Caveat: a USDT-quoted coin's "return" and "volume" are relative to USDT,
# so they wobble a little if USDT itself is depegging at the same time.
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

BINANCE_KLINE_COLS = [
    "open_time", "open", "high", "low", "close", "volume", "close_time",
    "quote_volume", "trades", "taker_buy_base_volume", "taker_buy_quote_volume", "ignore",
]

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
DEPEG_GAP_HOURS = 24  # Inactivity gap to merge continuous ongoing crash
FORECAST_HORIZONS_HOURS = [1, 6, 24]

# ---------------------------------------------------------------------------
# On-chain feature parameters
# ---------------------------------------------------------------------------
WHALE_USD_THRESHOLD = 1_000_000
DROP_SELF_TRANSFERS = True
OUTLIER_REVIEW_THRESHOLD = 50_000_000

# ---------------------------------------------------------------------------
# Leakage control
# ---------------------------------------------------------------------------
EMBARGO_HOURS = 24
# train = hours before this (minus the horizon), test = hours after it.
# 2023-01-01 -> ~59% of rows in train, 22 of 105 episodes (5 coins) in test
SPLIT_CUTOFF = "2023-01-01"