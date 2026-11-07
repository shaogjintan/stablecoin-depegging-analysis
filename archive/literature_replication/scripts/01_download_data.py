#!/usr/bin/env python3
"""
TASK 1: Download and clean price data (Lee et al. 2025 replication).
- CoinGecko keyless API: hourly prices + volumes for 7 coins, 2022-04-01 .. 2022-06-30 UTC.
- AlternativeMe Fear & Greed index for the same window.
- Saves data/may_2022_hourly_prices.csv and data/fear_greed_index.csv
"""
import time
import sys
import json
import datetime as dt

import pandas as pd
import requests

BASE = "https://api.coingecko.com/api/v3"
FROM = 1648771200      # 2022-04-01 00:00 UTC
TO   = 1656633599      # 2022-06-30 23:59 UTC

COINS = {
    "UST":  "terrausd",
    "USDT": "tether",
    "USDC": "usd-coin",
    "DAI":  "dai",
    "BUSD": "binance-usd",
    "BTC":  "bitcoin",
    "ETH":  "ethereum",
}

# Keyless tier: 5 calls/min. 15 s between calls keeps us safely under it.
SLEEP = 15.0


def fetch_market_chart(coin_id: str, frm: int, to: int, interval: str = "hourly"):
    url = f"{BASE}/coins/{coin_id}/market_chart/range"
    params = {"vs_currency": "usd", "from": frm, "to": to, "interval": interval}
    last_err = None
    for attempt in range(6):
        try:
            r = requests.get(url, params=params, timeout=60)
        except requests.RequestException as e:
            last_err = e
            time.sleep(5 * (attempt + 1))
            continue
        if r.status_code == 200:
            return r.json()
        if r.status_code == 429:
            wait = 20 * (attempt + 1)
            print(f"  429 rate-limited, waiting {wait}s ...", flush=True)
            time.sleep(wait)
            continue
        last_err = RuntimeError(f"HTTP {r.status_code}: {r.text[:200]}")
        # 4xx other than 429: maybe interval rejected -> retry without interval once
        if r.status_code == 400 and interval:
            interval = None
            print("  400 with interval -> retrying without interval (may fall to daily!)", flush=True)
            continue
        break
    raise last_err if last_err else RuntimeError("unknown failure")


def rows_for_coin(name: str, cg_id: str):
    data = fetch_market_chart(cg_id, FROM, TO, "hourly")
    prices = data.get("prices", [])
    volumes = data.get("total_volumes", [])
    if not prices:
        raise RuntimeError(f"{name}: empty prices array")
    p = pd.DataFrame(prices, columns=["ts", "price"]).set_index("ts")
    v = pd.DataFrame(volumes, columns=["ts", "volume"]).set_index("ts")
    df = p.join(v, how="outer")
    df["coin"] = name
    df.index = pd.to_datetime(df.index, unit="ms", utc=True)
    df = df[~df.index.duplicated(keep="first")].sort_index()
    return df[["coin", "price", "volume"]]


def resample_hourly(df: pd.DataFrame):
    """Force exact hourly UTC grid; forward-fill gaps (up to a tolerance)."""
    grid = pd.date_range(df.index.min().floor("h"), df.index.max().ceil("h"), freq="h", tz="UTC")
    return df.reindex(grid, method="ffill").dropna(subset=["price"])


def main():
    t0 = time.time()
    frames = {}
    for name, cg_id in COINS.items():
        print(f"[{dt.datetime.now():%H:%M:%S}] fetching {name} ({cg_id}) ...", flush=True)
        raw = rows_for_coin(name, cg_id)
        frames[name] = resample_hourly(raw)
        n = len(frames[name])
        print(f"  -> {n} hourly rows, "
              f"{frames[name].index.min():%Y-%m-%d %H:%M} .. {frames[name].index.max():%Y-%m-%d %H:%M}", flush=True)
        time.sleep(SLEEP)

    df = pd.concat(frames.values())
    df = df.reset_index().rename(columns={"index": "timestamp"})
    df = df.sort_values(["coin", "timestamp"]).reset_index(drop=True)
    out = "data/may_2022_hourly_prices.csv"
    df.to_csv(out, index=False)
    print(f"saved {out}: {len(df)} rows")

    # ---- Fear & Greed ----
    print("fetching Fear & Greed (alternative.me) ...", flush=True)
    r = requests.get("https://api.alternative.me/fng/", params={"limit": 0, "format": "json"}, timeout=60)
    r.raise_for_status()
    fng = pd.DataFrame(r.json()["data"])
    fng["timestamp"] = pd.to_datetime(fng["timestamp"].astype(int), unit="s", utc=True)
    fng["value"] = fng["value"].astype(int)
    fng = fng[(fng["timestamp"] >= "2022-04-01") & (fng["timestamp"] <= "2022-07-01")]
    fng = fng.sort_values("timestamp")[["timestamp", "value", "value_classification"]]
    fout = "data/fear_greed_index.csv"
    fng.to_csv(fout, index=False)
    print(f"saved {fout}: {len(fng)} rows")

    # ---- summary ----
    print("\n===== SUMMARY =====")
    for name in COINS:
        sub = df[df["coin"] == name]
        print(f"{name:>4}: {len(sub):>5} rows | {sub['timestamp'].min()} .. {sub['timestamp'].max()} | "
              f"price range [{sub['price'].min():.4f}, {sub['price'].max():.4f}]")
    print("\nFirst 5 rows:")
    print(df.head().to_string(index=False))
    print(f"\nTotal runtime {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
