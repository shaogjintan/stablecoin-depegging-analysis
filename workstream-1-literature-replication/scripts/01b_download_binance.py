#!/usr/bin/env python3
"""
TASK 1 (fallback data route): CoinGecko keyless rejects >365-day history (HTTP 401,
error 10012), so use free public sources instead:
  - Binance data.binance.vision monthly 1h klines (no key) for
    USTUSDT, USDCUSDT, BUSDUSDT, BTCUSDT, ETHUSDT.
  - Coinbase Exchange public candles (no key) for DAI-USD and USDT-USD hourly
    (true-USD numeraire).
  - AlternativeMe Fear & Greed index.
Warmup months Jan-Mar 2022 are included so 30-day features are valid from 2022-04-01.
Saves:
  data/hourly_prices_full.csv    (2022-01-01 .. 2022-06-30, warmup + study window)
  data/may_2022_hourly_prices.csv(2022-04-01 .. 2022-06-30, exactly per spec)
  data/fear_greed_index.csv
"""
import io
import time
import zipfile

import pandas as pd
import requests

MONTHS = ["2022-01", "2022-02", "2022-03", "2022-04", "2022-05", "2022-06"]
PAIRS = {
    "UST":  "USTUSDT",
    "USDC": "USDCUSDT",
    "BUSD": "BUSDUSDT",
    "BTC":  "BTCUSDT",
    "ETH":  "ETHUSDT",
}
COINBASE_PAIRS = {"DAI": "DAI-USD", "USDT": "USDT-USD"}
KLINES_COLS = ["open_time", "open", "high", "low", "close", "volume",
               "close_time", "quote_volume", "trades", "taker_base",
               "taker_quote", "ignore"]


def fetch_binance_month(pair: str, month: str):
    url = (f"https://data.binance.vision/data/spot/monthly/klines/{pair}/1h/"
           f"{pair}-1h-{month}.zip")
    r = requests.get(url, timeout=120)
    if r.status_code == 404:
        print(f"  note: {pair} {month} archive missing (delisted?) - skipping", flush=True)
        return None
    if r.status_code != 200:
        raise RuntimeError(f"Binance {pair} {month}: HTTP {r.status_code}")
    with zipfile.ZipFile(io.BytesIO(r.content)) as z:
        name = z.namelist()[0]
        raw = z.read(name).decode("utf-8")
    df = pd.read_csv(io.StringIO(raw), header=None, names=KLINES_COLS)
    df["timestamp"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df["price"] = df["close"].astype(float)
    df["volume"] = df["quote_volume"].astype(float)  # quote (USDT ~ USD) volume
    return df[["timestamp", "price", "volume"]]


def fetch_kraken(kpair: str):
    """Kraken public 60-min OHLC, paginated 720 candles per call, true-USD quote."""
    rows = []
    start_ts = int(pd.Timestamp("2022-01-01", tz="UTC").timestamp())
    end_ts = int(pd.Timestamp("2022-07-01", tz="UTC").timestamp())
    since = start_ts
    while True:
        params = {"pair": kpair, "interval": 60, "since": since}
        r = requests.get("https://api.kraken.com/0/public/OHLC", params=params, timeout=60)
        r.raise_for_status()
        js = r.json()
        if js.get("error"):
            raise RuntimeError(f"Kraken {kpair} error: {js['error']}")
        result = js["result"]
        key = kpair if kpair in result else next(k for k in result if k != "last")
        candles = result[key]
        last = int(result["last"])
        for c in candles:
            t = pd.to_datetime(int(c[0]), unit="s", utc=True)
            if t >= pd.Timestamp("2022-07-01", tz="UTC"):
                continue
            if t < pd.Timestamp("2022-01-01", tz="UTC"):
                continue
            rows.append({"timestamp": t, "price": float(c[4]),
                         "volume": float(c[6])})  # close, quote volume
        if last >= end_ts or len(candles) < 720:
            break
        since = last
        time.sleep(2)
    return pd.DataFrame(rows)


def fetch_coinbase_range(product: str, start_s: str, end_s: str):
    """Coinbase Exchange public candles for an arbitrary window (300/hr per call)."""
    rows = []
    start = pd.Timestamp(start_s, tz="UTC")
    end = pd.Timestamp(end_s, tz="UTC")
    while start < end:
        chunk_end = min(start + pd.Timedelta(hours=300), end)
        params = {"granularity": 3600,
                  "start": start.strftime("%Y-%m-%dT%H:%M:%SZ"),
                  "end": chunk_end.strftime("%Y-%m-%dT%H:%M:%SZ")}
        r = requests.get(f"https://api.exchange.coinbase.com/products/{product}/candles",
                         params=params, timeout=60)
        r.raise_for_status()
        for c in r.json():
            rows.append({"timestamp": pd.to_datetime(int(c[0]), unit="s", utc=True),
                         "price": float(c[4]),
                         "volume": float(c[5])})
        start = chunk_end
        time.sleep(0.4)
    return pd.DataFrame(rows)


def fetch_coinbase(product: str):
    """Coinbase Exchange public candles, 300/hourly per call, paginated start->end."""
    rows = []
    start = pd.Timestamp("2022-01-01", tz="UTC")
    end = pd.Timestamp("2022-07-01", tz="UTC")
    while start < end:
        chunk_end = min(start + pd.Timedelta(hours=300), end)
        params = {"granularity": 3600,
                  "start": start.strftime("%Y-%m-%dT%H:%M:%SZ"),
                  "end": chunk_end.strftime("%Y-%m-%dT%H:%M:%SZ")}
        r = requests.get(f"https://api.exchange.coinbase.com/products/{product}/candles",
                         params=params, timeout=60)
        r.raise_for_status()
        candles = r.json()  # [[time, low, high, open, close, volume], ...] newest first
        for c in candles:
            rows.append({"timestamp": pd.to_datetime(int(c[0]), unit="s", utc=True),
                         "price": float(c[4]),
                         "volume": float(c[5])})
        start = chunk_end
        time.sleep(0.4)
    return pd.DataFrame(rows)


def main():
    frames = {}
    for coin, pair in PAIRS.items():
        print(f"fetching Binance {pair} (6 monthly zips) ...", flush=True)
        parts = [p for p in (fetch_binance_month(pair, m) for m in MONTHS) if p is not None]
        if not parts:
            raise RuntimeError(f"{pair}: no monthly archives found")
        frames[coin] = pd.concat(parts, ignore_index=True)
        time.sleep(1)

    for coin, product in COINBASE_PAIRS.items():
        print(f"fetching Coinbase Exchange {product} ...", flush=True)
        cb = fetch_coinbase(product)
        if cb.empty:
            raise RuntimeError(f"Coinbase {product}: no candles returned")
        frames[coin] = cb

    full = []
    for coin, df in frames.items():
        df = df.sort_values("timestamp").drop_duplicates("timestamp")
        df.insert(0, "coin", coin)
        full.append(df)
    full = pd.concat(full, ignore_index=True)
    full = full[full["timestamp"] < pd.Timestamp("2022-07-01", tz="UTC")]
    full.to_csv("data/hourly_prices_full.csv", index=False)
    print(f"saved data/hourly_prices_full.csv: {len(full)} rows")

    may = full[(full["timestamp"] >= "2022-04-01") & (full["timestamp"] < "2022-07-01")]
    may.to_csv("data/may_2022_hourly_prices.csv", index=False)
    print(f"saved data/may_2022_hourly_prices.csv: {len(may)} rows")

    # ---- Fear & Greed ----
    print("fetching Fear & Greed (alternative.me) ...", flush=True)
    r = requests.get("https://api.alternative.me/fng/",
                     params={"limit": 0, "format": "json"}, timeout=60)
    r.raise_for_status()
    fng = pd.DataFrame(r.json()["data"])
    fng["timestamp"] = pd.to_datetime(fng["timestamp"].astype(int), unit="s", utc=True)
    fng["value"] = fng["value"].astype(int)
    fng = fng[(fng["timestamp"] >= "2022-04-01") & (fng["timestamp"] < "2022-07-01")]
    fng = fng.sort_values("timestamp")[["timestamp", "value", "value_classification"]]
    fng.to_csv("data/fear_greed_index.csv", index=False)
    print(f"saved data/fear_greed_index.csv: {len(fng)} rows")

    print("\n===== SUMMARY (Apr-Jun study window) =====")
    for coin in ["UST", "USDT", "USDC", "DAI", "BUSD", "BTC", "ETH"]:
        sub = may[may["coin"] == coin]
        if len(sub) == 0:
            print(f"{coin:>4}: NO DATA")
            continue
        print(f"{coin:>4}: {len(sub):>5} rows | {sub['timestamp'].min()} .. "
              f"{sub['timestamp'].max()} | price range "
              f"[{sub['price'].min():.4f}, {sub['price'].max():.4f}]")
    print("\nFirst 5 rows:")
    print(may.head().to_string(index=False))


if __name__ == "__main__":
    main()
