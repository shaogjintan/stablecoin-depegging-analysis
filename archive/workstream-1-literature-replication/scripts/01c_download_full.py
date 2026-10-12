#!/usr/bin/env python3
"""Extend the dataset to the paper's full window: 2022-01-01 .. 2023-12-31."""
import importlib.util
import time

import pandas as pd

spec = importlib.util.spec_from_file_location("dl", "scripts/01b_download_binance.py")
dl = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dl)

MONTHS = [f"{y}-{m:02d}" for y in (2022, 2023) for m in range(1, 13)]
PAIRS = {
    "UST":  "USTUSDT",
    "USDC": "USDCUSDT",
    "BUSD": "BUSDUSDT",
    "BTC":  "BTCUSDT",
    "ETH":  "ETHUSDT",
}

frames = {}
for coin, pair in PAIRS.items():
    print(f"fetching Binance {pair} ({len(MONTHS)} monthly zips) ...", flush=True)
    parts = [p for p in (dl.fetch_binance_month(pair, m) for m in MONTHS) if p is not None]
    if not parts:
        raise RuntimeError(f"{pair}: no monthly archives found")
    frames[coin] = pd.concat(parts, ignore_index=True)
    time.sleep(0.5)

for coin, product in {"DAI": "DAI-USD", "USDT": "USDT-USD"}.items():
    print(f"fetching Coinbase {product} (2022-2023) ...", flush=True)
    frames[coin] = dl.fetch_coinbase_range(product, "2022-01-01", "2024-01-01")

full = []
for coin, df in frames.items():
    df = df.sort_values("timestamp").drop_duplicates("timestamp")
    df.insert(0, "coin", coin)
    full.append(df)
full = pd.concat(full, ignore_index=True)
full = full[(full["timestamp"] >= "2022-01-01") & (full["timestamp"] < "2024-01-01")]
full.to_csv("data/hourly_prices_full_2022_2023.csv", index=False)
print(f"saved data/hourly_prices_full_2022_2023.csv: {len(full)} rows")
for coin in ["UST", "USDT", "USDC", "DAI", "BUSD", "BTC", "ETH"]:
    sub = full[full["coin"] == coin]
    print(f"  {coin:>4}: {len(sub):>6} rows | {sub['timestamp'].min().date()} .. {sub['timestamp'].max().date()}")
