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
DUNE_ONCHAIN_DIR = RAW_DATA_DIR / "dune_onchain"
PROCESSED_DATA_DIR = Path("data/processed")
REPORT_DIR = Path("reports")

for d in [PROCESSED_DATA_DIR, REPORT_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Universe
# ---------------------------------------------------------------------------
COINS = [
    "USDT", "USDC", "DAI", "UST", "PAX",
    "USDD", "MIM", "USDN", "TUSD", "GUSD",
    "FEI", "IRON", "LUSD", "crvUSD", "USDe",
    "FDUSD", "FRAX",
    "BUSD", "GHO", "PYUSD", "USDS",
]

AUXILIARY_COINS = ["WLUNA", "MKR", "CRV"]
ALL_COINS = COINS + AUXILIARY_COINS

# ERC-20 contract addresses (Ethereum mainnet)
CONTRACT_ADDRESSES = {
    "USDT": "0xdac17f958d2ee523a2206206994597c13d831ec7",
    "USDC": "0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48",
    "DAI": "0x6b175474e89094c44da98b954eedeac495271d0f",
    "UST": "0xa693b19d2931d498c5b318df961919bb4aee87a5",
    "PAX": "0x8e870d67f660d95d5be530380d0ec0bd388289e1",
    "WLUNA": "0xd2877702675e6ceb975b4a1dff9fb7baf4c91ea9",
    "MKR": "0x9f8f72aa9304c8b593d555f12ef6589cc3a579a2",
    "CRV": "0xd533a949740bb3306d119cc777fa900ba034cd52",
    "USDD": "0x0c10bf8fcb7bf5412187a595ab01a96c20cf4b7a", #0x4f8e5DE400DE08B164E7421B3EE387f461beCD1A
    "MIM": "0x99d8a9c45b2eca8864373a26d1459e3dff1e17f3",
    "USDN": "0x674c6ad92fd080e4004b2312b45f796a192d27a0",
    "TUSD": "0x0000000000085d4780b73119b644ae5ecd22b376",
    "GUSD": "0x056fd409e1d7a124bd7017459dfea2f387b6d5cd",
    "FEI": "0x956f47f50a910163d8bf957cf5846d573e7f87ca",
    "IRON": "0xd86b5923f3ad7b585ed81b448170ae026c65ae9a",
    "LUSD": "0x5f98805a4e8be255a32880fdec7f6728c6568ba0",
    "crvUSD": "0xf939e0a03fb07f59a73314e73794be0e57ac1b4e",
    "USDe": "0x4c9edd5852cd905f086c759e8383e09bff1e68b3",
    "FDUSD": "0xc5f0f7b66764f6ec8c8dff7ba683102295e16409",
    "FRAX": "0x853d955acef822db058eb8505911ed77f175b99e",
    "BUSD": "0x4fabb145d64652a948d72533023f6e7a623c7c53",
    "GHO": "0x40d16fc0246ad3160ccc09b8d0d3a2cd28ae6c2f",
    "PYUSD": "0x6c3ea9036406852006290770bedfcaba0e23a0e8",
    "USDS": "0xdc035d45d973e3ec169d2276ddab16f1e407384f",
}

TOKEN_DECIMALS = {
    "USDT": 6, "USDC": 6, "DAI": 18, "UST": 18,
    "USDD": 18, "MIM": 18, "USDN": 18, "TUSD": 18,
    "GUSD": 2, "FEI": 18, "IRON": 18, "LUSD": 18,
    "crvUSD": 18, "USDe": 18, "FDUSD": 18, "FRAX": 18,
    "WLUNA": 18, "MKR": 18, "CRV": 18, "PAX": 18,
    "BUSD": 18, "GHO": 18, "PYUSD": 6, "USDS": 18,
}

# ---------------------------------------------------------------------------
# Study window
# ---------------------------------------------------------------------------
START_DATE = "2018-01-01"
END_DATE = "2025-12-31"

# ---------------------------------------------------------------------------
# Depeg definition
# ---------------------------------------------------------------------------
DEPEG_BAND = 0.01
DEPEG_MIN_DURATION_HOURS = 2
FORECAST_HORIZONS_HOURS = [1, 6, 24]
LOOKBACK_WINDOWS_HOURS = [6, 24, 48]

# ---------------------------------------------------------------------------
# DEX pool data sources
# ---------------------------------------------------------------------------
CURVE_API_POOLS_URL = "https://api.curve.fi/api/getPools/all/ethereum"

# DeFiLlama yields API for historical TVL
DEFILLAMA_POOLS_URL = "https://yields.llama.fi/pools"
DEFILLAMA_POOL_CHART_URL = "https://yields.llama.fi/chart/{pool_id}"

# Curve pool addresses for all stablecoins
# Source: Curve documentation and Etherscan
DEX_POOL_ADDRESSES = {
    "curve_3pool": "0xbebc44782c7db0a1a60cb6fe97d0b483032ff1c7",           # DAI/USDC/USDT
    "curve_ust_3pool": "0x890f4e345b1daed0367a877a1612f86a1f86985f",     # UST/3Crv metapool
    "curve_frax_usdc": "0xdcef968d416a41cdac0ed8702fac8128a64241a2",     # FRAX/USDC
    "curve_lusd_usdc": "0x6c5a90b7bb8d17e3b9df6da73edda83dc5b2bfd9",     # LUSD/USDC
    "curve_mim_3pool": "0x5a6a4d54456819380173272a5e8e9b9904bdf41b",     # MIM/3Crv
    "curve_usdd_3pool": "0xb7f748f3d78efc312195a570420a1406546d662e",    # USDD/3Crv
    "curve_tusd_3pool": "0x6de449cffa49f4d263761f3af63a0c3bba1306ba",    # TUSD/3Crv
    "curve_gusd_3pool": "0x4f062658eaaf2c1ccf8c8e36d6824cdf41167956",    # GUSD/3Crv
    "curve_fei_3pool": "0x906c62cd4cf422ba4e3730b34ff41b82adaa196d",     # FEI/3Crv
    "curve_pax_3pool": "0x4b8cf1964bdd37f0ce4900e2a0f55d44de3158fe",     # PAX/3Crv (PAX rebranded to USDP)
}

# DeFiLlama pool IDs for TVL history
# These will be populated dynamically by 01e_fetch_all_data.py
# But we provide known IDs for major pools as fallback
DEFILLAMA_POOL_IDS = {
    "curve_3pool": "",           # Discover current DeFiLlama Yields chart ID dynamically
    "curve_ust_3pool": "",      # Will be discovered
    "curve_frax_usdc": "",      # Will be discovered
    "curve_lusd_usdc": "",      # Will be discovered
    "curve_mim_3pool": "",      # Will be discovered
    "curve_usdd_3pool": "",     # Will be discovered
    "curve_tusd_3pool": "",     # Will be discovered
    "curve_gusd_3pool": "",     # Will be discovered
    "curve_fei_3pool": "",      # Will be discovered
    "curve_pax_3pool": "",      # Will be discovered
}

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