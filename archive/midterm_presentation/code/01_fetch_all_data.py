#!/usr/bin/env python3
"""
01e_fetch_all_data.py
Consolidated data fetching for all coins and DEX pool data.

OUTPUTS:
  - data/processed/hourly_price.parquet (all coins, hourly OHLC-like)
  - data/processed/dex_liquidity_features.parquet (pool TVL over time)
  - data/processed/curve_pool_snapshots.parquet (current pool state for reference)
"""

import json
import time
import datetime as dt
import pandas as pd
import requests
from pathlib import Path
from typing import Optional, Dict, List

from config import (
    PROCESSED_DATA_DIR,
    START_DATE,
    END_DATE,
    ALL_COINS,
    AUXILIARY_COINS,
    COINS,
    CURVE_API_POOLS_URL,
    DEFILLAMA_POOL_CHART_URL,
    DEFILLAMA_POOLS_URL,
    DEX_POOL_ADDRESSES,
    DEFILLAMA_POOL_IDS,
)

PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_END = END_DATE if END_DATE else dt.datetime.now().strftime("%Y-%m-%d")

SESSION = requests.Session()
SESSION.headers.update({
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json",
})
try:
    from urllib3.util.retry import Retry
    from requests.adapters import HTTPAdapter

    _retry = Retry(
        total=3,
        backoff_factor=1.0,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
    )
    SESSION.mount("https://", HTTPAdapter(max_retries=_retry))
except Exception:
    pass  # fall back to the per-call retry loops already used below

POOL_ID_CACHE_PATH = PROCESSED_DATA_DIR / "defillama_pool_id_cache.json"


def load_pool_id_cache() -> Dict[str, str]:
    if POOL_ID_CACHE_PATH.exists():
        try:
            with open(POOL_ID_CACHE_PATH, "r") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def save_pool_id_cache(cache: Dict[str, str]) -> None:
    try:
        with open(POOL_ID_CACHE_PATH, "w") as f:
            json.dump(cache, f, indent=2)
    except Exception as e:
        print(f"  Could not write pool ID cache: {e}")

DEFILLAMA_COIN_SPECS = {
    "USDT": {"id": "coingecko:tether", "start": "2018-01-01"},
    "USDC": {"id": "coingecko:usd-coin", "start": "2018-10-01"},
    "DAI": {"id": "coingecko:dai", "start": "2019-11-18"},
    "UST": {"id": "coingecko:terrausd", "start": "2020-09-01", "end": "2022-06-01"},
    "PAX": {"id": "coingecko:paxos-standard", "start": "2018-09-01", "end": "2024-01-01"},
    "USDD": {"id": "coingecko:usdd", "start": "2022-05-05"},
    "MIM": {"id": "coingecko:magic-internet-money", "start": "2021-08-15"},
    "USDN": {"id": "coingecko:neutrino", "start": "2020-06-01", "end": "2023-01-01"},
    "TUSD": {"id": "coingecko:true-usd", "start": "2018-03-01"},
    "GUSD": {"id": "coingecko:gemini-dollar", "start": "2018-09-01"},
    "FEI": {"id": "coingecko:fei-usd", "start": "2021-04-03", "end": "2022-12-31"},
    "IRON": {"id": "polygon:0xd86b5923f3ad7b585ed81b448170ae026c65ae9a", "start": "2021-06-01", "end": "2021-06-25"},
    "LUSD": {"id": "coingecko:liquity-usd", "start": "2021-04-05"},
    "crvUSD": {"id": "coingecko:crvusd", "start": "2023-06-01"},
    "USDe": {"id": "coingecko:ethena-usde", "start": "2024-01-01"},
    "FDUSD": {"id": "coingecko:first-digital-usd", "start": "2023-08-01"},
    "FRAX": {"id": "coingecko:frax", "start": "2021-01-01"},
    "WLUNA": {"id": "coingecko:terra-luna", "start": "2021-01-01", "end": "2022-06-01"},
    "MKR": {"id": "coingecko:maker", "start": "2018-01-01"},
    "CRV": {"id": "coingecko:curve-dao-token", "start": "2020-08-01"},
    "BUSD": {"id": "coingecko:binance-usd", "start": "2019-09-01", "end": "2024-02-29"},
    "GHO": {"id": "coingecko:gho", "start": "2023-07-01"},
    "PYUSD": {"id": "coingecko:paypal-usd", "start": "2023-08-01"},
    "USDS": {"id": "coingecko:usds", "start": "2024-09-01"},
}

POOL_COIN_MAPPING = {
    "curve_3pool": ["DAI", "USDC", "USDT"],
    "curve_ust_3pool": ["UST"],
    "curve_frax_usdc": ["FRAX", "USDC"],
    "curve_lusd_usdc": ["LUSD", "USDC"],
    "curve_mim_3pool": ["MIM"],
    "curve_usdd_3pool": ["USDD"],
    "curve_tusd_3pool": ["TUSD"],
    "curve_gusd_3pool": ["GUSD"],
    "curve_fei_3pool": ["FEI"],
    "curve_pax_3pool": ["PAX"],
}


def fetch_defillama_hourly(
    coin_name: str,
    coin_id: str,
    start_s: str,
    end_s: str,
    chunk_hours: int = 400
) -> pd.DataFrame:
    """Fetch hourly historical price points from DefiLlama chart API."""
    start_ts = int(pd.Timestamp(start_s, tz="UTC").timestamp())
    end_ts = int(pd.Timestamp(end_s, tz="UTC").timestamp())
    cur_ts = start_ts
    rows = []
    
    total_hours = max(1, (end_ts - start_ts) // 3600)
    print(f"\n[{dt.datetime.now():%H:%M:%S}] Fetching {coin_name:<6} ({coin_id}) {start_s} -> {end_s} (~{total_hours:,} hrs)...", flush=True)
    
    calls = 0
    consecutive_empty = 0
    while cur_ts < end_ts:
        hours_left = int((end_ts - cur_ts) // 3600) + 1
        chunk = min(chunk_hours, hours_left)
        url = f"https://coins.llama.fi/chart/{coin_id}?start={cur_ts}&span={chunk}&period=1h"
        
        success = False
        for attempt in range(5):
            try:
                r = SESSION.get(url, timeout=25)
                if r.status_code == 200:
                    prices = r.json().get("coins", {}).get(coin_id, {}).get("prices", [])
                    success = True
                    break
                elif r.status_code in (429, 502, 503, 504):
                    time.sleep(2.0 * (attempt + 1))
                    continue
                else:
                    print(f"  HTTP {r.status_code} for {coin_name} at ts={cur_ts}")
                    break
            except requests.RequestException as e:
                time.sleep(2.0 * (attempt + 1))
                continue
        
        calls += 1
        if not success or not prices:
            consecutive_empty += 1
            if consecutive_empty > 10:
                print(f"  Reached end of available data for {coin_name} at {pd.to_datetime(cur_ts, unit='s', utc=True)}")
                break
            cur_ts += chunk * 3600
            time.sleep(0.1)
            continue
        
        consecutive_empty = 0
        for p in prices:
            rows.append({
                "coin": coin_name,
                "timestamp_raw": p["timestamp"],
                "price": float(p["price"]),
                "volume": float(p.get("volume", 0.0) or 0.0)
            })
        
        last_ret_ts = prices[-1]["timestamp"]
        if last_ret_ts >= cur_ts:
            cur_ts = last_ret_ts + 3600
        else:
            cur_ts += chunk * 3600
        
        time.sleep(0.12)
        if calls % 25 == 0:
            print(f"  ...progress: reached {pd.to_datetime(cur_ts, unit='s', utc=True):%Y-%m-%d %H:%M} ({len(rows):,} rows)", flush=True)
    
    if not rows:
        print(f"  WARNING: No data collected for {coin_name}!")
        return pd.DataFrame()
    
    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["timestamp_raw"], unit="s", utc=True).dt.round("h")
    df = df.sort_values("timestamp").drop_duplicates(subset=["timestamp"], keep="last")
    
    grid = pd.date_range(df["timestamp"].min(), df["timestamp"].max(), freq="h", tz="UTC")
    df = df.set_index("timestamp").reindex(grid)
    df["coin"] = coin_name
    df["price"] = df["price"].interpolate(method="time", limit=3).ffill().bfill()
    df["volume"] = df["volume"].fillna(0.0)
    df = df.reset_index().rename(columns={"index": "timestamp"})
    df = df[["coin", "timestamp", "price", "volume"]].dropna(subset=["price"])
    
    print(f"  -> SUCCESS {coin_name}: {len(df):,} hourly rows | {df['timestamp'].min():%Y-%m-%d %H:%M} .. {df['timestamp'].max():%Y-%m-%d %H:%M}")
    return df


def fetch_defillama_pool_ids() -> Dict[str, str]:
    """
    Discover DeFiLlama Yields chart IDs for the configured Curve pools.
    """
    try:
        resp = SESSION.get(DEFILLAMA_POOLS_URL, timeout=60)
        if resp.status_code != 200:
            print(f"  DeFiLlama API returned {resp.status_code}")
            return {}

        payload = resp.json()
        pools = payload.get("data", []) if isinstance(payload, dict) else payload
        if not isinstance(pools, list) or not pools:
            print("  No pools found in DeFiLlama response")
            return {}

        pool_id_map = {}

        # Build normalized target metadata.
        targets = {
            name: {
                "address": (addr or "").lower(),
                "keywords": [
                    x for x in name.replace("curve_", "").lower().split("_")
                    if x
                ],
            }
            for name, addr in DEX_POOL_ADDRESSES.items()
        }

        def nested_strings(obj):
            """Yield strings from shallow/nested dict/list metadata."""
            if isinstance(obj, str):
                yield obj
            elif isinstance(obj, dict):
                for v in obj.values():
                    yield from nested_strings(v)
            elif isinstance(obj, list):
                for v in obj:
                    yield from nested_strings(v)

        # First pass: exact address matches.
        for pool in pools:
            if not isinstance(pool, dict):
                continue
            if str(pool.get("chain", "")).lower() not in ("ethereum", "eth"):
                continue

            pool_id = pool.get("poolId") or pool.get("pool")
            if not pool_id:
                continue

            text_blob = " ".join(
                str(s).lower() for s in nested_strings(pool)
            )

            for our_name, meta in targets.items():
                if meta["address"] and meta["address"] in text_blob:
                    pool_id_map[our_name] = str(pool_id)
                    print(
                        f"  Found DeFiLlama ID for {our_name}: {pool_id} "
                        f"(exact address)"
                    )

        # Second pass: exact/strong name matching, only if unique enough.
        for pool in pools:
            if not isinstance(pool, dict):
                continue
            if str(pool.get("chain", "")).lower() not in ("ethereum", "eth"):
                continue

            pool_id = pool.get("poolId") or pool.get("pool")
            if not pool_id:
                continue

            haystack = " ".join([
                str(pool.get("project", "")),
                str(pool.get("symbol", "")),
                str(pool.get("poolMeta", "")),
                str(pool.get("protocol", "")),
            ]).lower()

            for our_name, meta in targets.items():
                if our_name in pool_id_map:
                    continue

                # Require Curve + at least one meaningful pool-specific keyword.
                if "curve" not in haystack:
                    continue

                keywords = [k for k in meta["keywords"] if len(k) >= 3]
                score = sum(k in haystack for k in keywords)

                if score >= max(1, min(2, len(keywords))):
                    pool_id_map[our_name] = str(pool_id)
                    print(
                        f"  Found DeFiLlama ID for {our_name}: {pool_id} "
                        f"(name/symbol fallback)"
                    )
                    break

        return pool_id_map

    except Exception as e:
        print(f"Error fetching DeFiLlama pool IDs: {e}")
        return {}

def fetch_defillama_pool_tvl(pool_id: str, start_s: str, end_s: str) -> pd.DataFrame:
    """Fetch historical TVL for a pool from DeFiLlama."""
    if not pool_id:
        return pd.DataFrame()
    
    url = DEFILLAMA_POOL_CHART_URL.format(pool_id=pool_id)
    
    try:
        resp = SESSION.get(url, timeout=30)
        if resp.status_code != 200:
            return pd.DataFrame()
        
        data = resp.json()
        tvl_data = data.get("data", [])
        if not tvl_data:
            return pd.DataFrame()
        
        df = pd.DataFrame(tvl_data)
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s", utc=True)
        df = df[(df["timestamp"] >= pd.Timestamp(start_s, tz="UTC")) &
                (df["timestamp"] <= pd.Timestamp(end_s, tz="UTC"))]
        df = df.sort_values("timestamp")
        return df[["timestamp", "tvlUsd"]]
    except Exception as e:
        print(f"  TVL fetch failed: {e}")
        return pd.DataFrame()


def _extract_curve_pools(data) -> List[dict]:
    """Normalize the different response envelopes used by Curve's API."""
    if isinstance(data, list):
        return data

    if not isinstance(data, dict):
        return []

    candidates = [
        data.get("poolData"),
        data.get("data", {}).get("poolData") if isinstance(data.get("data"), dict) else None,
        data.get("data") if isinstance(data.get("data"), list) else None,
        data.get("pools"),
        data.get("result"),
    ]
    for candidate in candidates:
        if isinstance(candidate, list):
            return candidate
    return []


def _pool_tvl_usd(pool: dict) -> float:
    """Read Curve's current USD TVL field across API schema versions."""
    for key in ("usdTotal", "tvl", "total_liquidity", "totalLiquidityUSD"):
        value = pool.get(key)
        if value is not None:
            try:
                return float(value)
            except (TypeError, ValueError):
                pass
    return 0.0


def _pool_tokens(pool: dict) -> list:
    """Return token symbols from Curve's current `coins` field, with legacy fallbacks."""
    raw = pool.get("coins")
    if raw is None:
        raw = pool.get("tokens", [])

    symbols = []
    if isinstance(raw, list):
        for token in raw:
            if isinstance(token, dict):
                symbol = (
                    token.get("symbol")
                    or token.get("name")
                    or token.get("coinSymbol")
                    or ""
                )
                symbols.append(str(symbol).upper())
            elif isinstance(token, str):
                symbols.append(token.upper())
    return symbols


def _pool_balances(pool: dict) -> list:
    """Extract balances from current/legacy Curve pool objects."""
    coins = pool.get("coins")
    if isinstance(coins, list):
        balances = []
        for coin in coins:
            if isinstance(coin, dict):
                balances.append(
                    coin.get("poolBalance",
                    coin.get("balance",
                    coin.get("pool_balance", 0)))
                )
        if balances:
            return balances

    return pool.get("balances", [])


def fetch_curve_pool_state(pool_address: str) -> Optional[dict]:
    """
    Fetch current Curve pool state.
    """
    try:
        resp = SESSION.get(CURVE_API_POOLS_URL, timeout=30)
        if resp.status_code != 200:
            return None

        pools = _extract_curve_pools(resp.json())
        target = pool_address.lower()

        for pool in pools:
            pool_addr = str(pool.get("address", "")).lower()
            if pool_addr != target:
                continue

            coins = pool.get("coins", [])
            tokens = _pool_tokens(pool)

            try:
                virtual_price = float(
                    pool.get("virtualPrice",
                    pool.get("virtual_price", 1.0)) or 1.0
                )
            except (TypeError, ValueError):
                virtual_price = 1.0

            return {
                "name": pool.get("name", pool.get("symbol", "")),
                "address": pool.get("address", pool_address),
                "tvl": _pool_tvl_usd(pool),
                "balances": _pool_balances(pool),
                "tokens": tokens,
                "coins": coins,
                "price_oracle": pool.get("price_oracle", []),
                "pool_type": pool.get(
                    "pool_type",
                    pool.get("implementation", pool.get("assetTypeName", ""))
                ),
                "virtual_price": virtual_price,
                "curve_pool_id": pool.get("id", ""),
                "registry_id": pool.get("registryId", ""),
                "is_meta_pool": bool(pool.get("isMetaPool", False)),
                "is_broken": bool(pool.get("isBroken", False)),
            }

        return None

    except Exception as e:
        print(f"  Curve API state fetch failed for {pool_address}: {e}")
        return None

def fetch_all_pools_from_api() -> List[dict]:
    """Fetch all current Curve pools relevant to our stablecoins."""
    try:
        resp = SESSION.get(CURVE_API_POOLS_URL, timeout=30)
        if resp.status_code != 200:
            print(f"  Curve API returned HTTP {resp.status_code}")
            return []

        pools = _extract_curve_pools(resp.json())
        if not pools:
            print("  Curve API returned no pool records")
            return []

        stablecoin_symbols = {c.upper() for c in COINS}
        filtered_pools = []

        for pool in pools:
            token_symbols = _pool_tokens(pool)
            pool_name = str(pool.get("name", "")).upper()
            pool_symbol = str(pool.get("symbol", "")).upper()

            matching_coins = [
                s for s in token_symbols if s in stablecoin_symbols
            ]

            for coin in stablecoin_symbols:
                if coin in pool_name or coin in pool_symbol:
                    if coin not in matching_coins:
                        matching_coins.append(coin)

            tvl = _pool_tvl_usd(pool)

            if matching_coins and tvl > 100000:
                filtered_pools.append({
                    "pool_name": pool.get("name", pool_symbol),
                    "pool_address": pool.get("address", ""),
                    "tvl_usd": tvl,
                    "tokens": token_symbols,
                    "balances": _pool_balances(pool),
                    "coins": pool.get("coins", []),
                    "pool_type": pool.get(
                        "pool_type",
                        pool.get("implementation", pool.get("assetTypeName", ""))
                    ),
                    "matching_coins": matching_coins,
                    "virtual_price": float(
                        pool.get("virtualPrice",
                        pool.get("virtual_price", 1.0)) or 1.0
                    ),
                    "curve_pool_id": pool.get("id", ""),
                    "registry_id": pool.get("registryId", ""),
                })

        return filtered_pools

    except Exception as e:
        print(f"Error fetching all pools: {e}")
        return []

def fetch_curve_pool_data() -> pd.DataFrame:
    """Fetch current state of all Curve pools relevant to our stablecoins."""
    print("\n" + "=" * 70)
    print("STEP 2: Fetching Curve pool states")
    print("=" * 70)
    
    all_pool_states = []
    
    print("Fetching all pools from Curve API...")
    all_pools = fetch_all_pools_from_api()
    
    if all_pools:
        print(f"Found {len(all_pools)} pools containing our stablecoins:")
        for pool in all_pools:
            print(f"  {pool['pool_name']}: TVL ${pool['tvl_usd']:,.2f}, Coins: {pool['matching_coins']}")
        all_pool_states.extend(all_pools)
    
    known_addresses = {
        str(p.get("pool_address", p.get("address", ""))).lower()
        for p in all_pool_states
    }

    for pool_name, pool_addr in DEX_POOL_ADDRESSES.items():
        if not pool_addr:
            continue

        if pool_addr.lower() in known_addresses:
            continue
        
        print(f"\nFetching state for {pool_name} ({pool_addr[:10]}...)")
        state = fetch_curve_pool_state(pool_addr)
        if state:
            state["pool_name"] = pool_name
            state["matching_coins"] = POOL_COIN_MAPPING.get(pool_name, [])

            # Avoid duplicate rows if automatic discovery found the same
            # physical pool under a different display name.
            all_pool_states.append(state)

            print(f"  TVL: ${state['tvl']:,.2f}")
            print(f"  Tokens: {state.get('tokens', [])}")
            if state.get("curve_pool_id"):
                print(f"  Curve ID: {state['curve_pool_id']}")
            if state.get("registry_id"):
                print(f"  Registry: {state['registry_id']}")
        else:
            print(
                f"  Pool is not present in Curve's current active-pool response "
                f"(likely inactive/retired): {pool_name}"
            )
    
    if not all_pool_states:
        print("\nNo Curve pool states retrieved.")
        return pd.DataFrame()
    
    df = pd.DataFrame(all_pool_states)
    
    out_path = PROCESSED_DATA_DIR / "curve_pool_snapshots.parquet"
    df.to_parquet(out_path, index=False)
    print(f"\nSaved {len(df)} Curve pool states -> {out_path}")
    
    out_csv = PROCESSED_DATA_DIR / "curve_pool_snapshots.csv"
    df.to_csv(out_csv, index=False)
    print(f"Saved {len(df)} Curve pool states -> {out_csv}")
    
    return df


def fetch_defillama_pool_tvl_data() -> pd.DataFrame:
    """Fetch historical TVL data for pools from DeFiLlama."""
    print("\n" + "=" * 70)
    print("STEP 3: Fetching DeFiLlama pool TVL data")
    print("=" * 70)
    
    pool_id_map = load_pool_id_cache()
    if pool_id_map:
        print(f"Loaded {len(pool_id_map)} cached DeFiLlama pool IDs from {POOL_ID_CACHE_PATH}")

    missing = [p for p in DEX_POOL_ADDRESSES if DEX_POOL_ADDRESSES[p] and p not in pool_id_map]
    if missing:
        print(f"Looking up DeFiLlama pool IDs for {len(missing)} uncached pool(s)...")
        discovered = fetch_defillama_pool_ids()
        pool_id_map.update(discovered)
        save_pool_id_cache(pool_id_map)
    else:
        print("All configured pools already have cached DeFiLlama IDs.")

    # Update the in-memory config dict too (used later in this function)
    if pool_id_map:
        for pool_name, pool_id in pool_id_map.items():
            if pool_name in DEFILLAMA_POOL_IDS:
                DEFILLAMA_POOL_IDS[pool_name] = pool_id
    
    all_tvl = []
    start_date = START_DATE if START_DATE else "2018-01-01"
    end_date = DEFAULT_END
    
    for pool_name, pool_id in DEFILLAMA_POOL_IDS.items():
        # If we don't have an ID yet, try to find it
        if not pool_id and pool_name in DEX_POOL_ADDRESSES:
            pool_addr = DEX_POOL_ADDRESSES[pool_name]
            print(f"\nLooking up DeFiLlama ID for {pool_name} ({pool_addr[:10]}...)")
            # Try to find it in the map
            if pool_name in pool_id_map:
                pool_id = pool_id_map[pool_name]
                DEFILLAMA_POOL_IDS[pool_name] = pool_id
            else:
                print(f"  No DeFiLlama ID found for {pool_name}")
                continue
        
        if not pool_id:
            print(f"\nNo pool ID for {pool_name} - skipping")
            continue
        
        print(f"\nFetching TVL for {pool_name} (ID: {pool_id})...")
        tvl_df = fetch_defillama_pool_tvl(pool_id, start_date, end_date)

        if tvl_df.empty and pool_name in DEX_POOL_ADDRESSES:
            print(
                "  Chart returned no rows. The configured DeFiLlama ID may be "
                "stale or the pool may no longer be in the Yields dataset."
            )
        
        if not tvl_df.empty:
            tvl_df["pool_name"] = pool_name
            all_tvl.append(tvl_df)
            print(f"  Retrieved {len(tvl_df)} TVL records")
        else:
            print(f"  No TVL data available for {pool_name}")
    
    if not all_tvl:
        print("\nNo TVL data retrieved.")
        return pd.DataFrame()
    
    df = pd.concat(all_tvl, ignore_index=True)
    df = df.sort_values(["pool_name", "timestamp"]).reset_index(drop=True)
    
    out_path = PROCESSED_DATA_DIR / "dex_liquidity_features.parquet"
    df.to_parquet(out_path, index=False)
    print(f"\nSaved {len(df)} TVL records -> {out_path}")
    
    out_csv = PROCESSED_DATA_DIR / "dex_liquidity_features.csv"
    df.to_csv(out_csv, index=False)
    print(f"Saved {len(df)} TVL records -> {out_csv}")
    
    return df


def build_ohlc_from_price(price_df: pd.DataFrame) -> pd.DataFrame:
    """Convert simple price series to OHLC-like format."""
    df = price_df.copy()
    df["open"] = df["price"]
    df["high"] = df["price"]
    df["low"] = df["price"]
    df["close"] = df["price"]
    return df


def fetch_all_coin_prices() -> pd.DataFrame:
    """Fetch price data for all coins from DefiLlama."""
    price_out_parquet = PROCESSED_DATA_DIR / "hourly_price.parquet"
    
    print("=" * 70)
    print("STEP 1: Fetching hourly price data for all coins")
    print("=" * 70)
    
    price_frames = []
    failed_coins = []
    end_date = DEFAULT_END
    start_date = START_DATE if START_DATE else "2018-01-01"
    
    for coin_name, spec in DEFILLAMA_COIN_SPECS.items():
        if coin_name not in ALL_COINS:
            continue
        
        coin_end = spec.get("end", end_date)
        coin_start = spec.get("start", start_date)
        
        df_coin = fetch_defillama_hourly(coin_name, spec["id"], coin_start, coin_end)
        
        if not df_coin.empty:
            ohlc_df = build_ohlc_from_price(df_coin)
            price_frames.append(ohlc_df)
        else:
            failed_coins.append(coin_name)
    
    if not price_frames:
        raise RuntimeError("No price data collected for any coin!")
    
    all_prices = pd.concat(price_frames, ignore_index=True)
    all_prices = all_prices.sort_values(["coin", "timestamp"]).reset_index(drop=True)
    
    try:
        all_prices[["coin", "timestamp", "open", "high", "low", "close", "volume"]].to_parquet(
            price_out_parquet, index=False
        )
        print(f"Saved {len(all_prices):,} price rows -> {price_out_parquet}")
    except Exception as e:
        print(f"Parquet save skipped: {e}")
    
    price_out_csv = PROCESSED_DATA_DIR / "hourly_price.csv"
    all_prices["timestamp_str"] = all_prices["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S+00:00")
    price_export = all_prices[["coin", "timestamp_str", "open", "high", "low", "close", "volume"]].rename(
        columns={"timestamp_str": "timestamp"}
    )
    price_export.to_csv(price_out_csv, index=False)
    print(f"Saved {len(price_export):,} price rows -> {price_out_csv}")
    
    if failed_coins:
        print(f"\nWARNING: Failed to fetch data for: {failed_coins}")
    
    return all_prices


def main():
    """Main execution - fetches all data."""
    t0 = time.time()
    
    print("=" * 70)
    print("STABLECOIN DEPEG DATA FETCHER")
    print(f"Study window: {START_DATE} to {END_DATE}")
    print(f"Coins: {len(COINS)} stablecoins + {len(AUXILIARY_COINS)} auxiliary")
    print(f"Curve pools: {len(DEX_POOL_ADDRESSES)}")
    print("=" * 70)
    
    # 1. Fetch price data
    price_df = fetch_all_coin_prices()
    print("\nPrice data summary:")
    print(price_df.groupby("coin")["timestamp"].agg(["min", "max", "count"]))
    
    # 2. Fetch Curve pool states
    pool_df = fetch_curve_pool_data()
    
    # 3. Fetch DeFiLlama TVL data
    tvl_df = fetch_defillama_pool_tvl_data()
    
    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    print(f"\nTotal elapsed time: {time.time() - t0:.1f}s")
    
    print("\n" + "=" * 70)
    print("COMPLETE!")
    print("=" * 70)


if __name__ == "__main__":
    main()