from __future__ import annotations

import io
import time
import datetime as dt
import pandas as pd
import requests
import zipfile

from config import (
    PROCESSED_DATA_DIR,
    START_DATE,
    END_DATE,
    FIAT_COINS,
    DEFILLAMA_FIAT_COIN_SPECS,
    EXCHANGE_OHLCV_SPECS,
    BINANCE_KLINE_COLS
)

PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_END = END_DATE if END_DATE else dt.datetime.now().strftime("%Y-%m-%d")

OUT_PATH = PROCESSED_DATA_DIR / "hourly_stablecoin.parquet"

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


def fetch_defillama_hourly(
    coin_name: str,
    coin_id: str,
    start_s: str,
    end_s: str,
    chunk_hours: int = 400
) -> pd.DataFrame:
    """Fetch hourly historical price points from DefiLlama chart API.

    Requests are kept on exact hour boundaries and use searchWidth=1800 (match
    the stored point nearest each hour within +/-30 min). DefiLlama's stored
    points sit a few minutes off the hour, so requests whose start drifts off
    :00 silently return only a handful of points per chunk."""
    start_ts = int(pd.Timestamp(start_s, tz="UTC").floor("h").timestamp())
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
        url = (f"https://coins.llama.fi/chart/{coin_id}?start={cur_ts}&span={chunk}"
               f"&period=1h&searchWidth=1800")

        success = False
        prices = []
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
            except requests.RequestException:
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
            })

        # fixed hour-aligned step (never re-anchor on the returned timestamps,
        # which drift off the hour)
        cur_ts += chunk * 3600

        time.sleep(0.12)
        if calls % 25 == 0:
            print(f"  ...progress: reached {pd.to_datetime(cur_ts, unit='s', utc=True):%Y-%m-%d %H:%M} ({len(rows):,} rows)", flush=True)

    if not rows:
        print(f"  WARNING: No data collected for {coin_name}!")
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["timestamp_raw"], unit="s", utc=True).dt.round("h")
    df = df[df["timestamp"] <= pd.Timestamp(end_s, tz="UTC")]  # rounding can spill past end
    df = df.sort_values("timestamp").drop_duplicates(subset=["timestamp"], keep="last")

    grid = pd.date_range(df["timestamp"].min(), df["timestamp"].max(), freq="h", tz="UTC")
    df = df.set_index("timestamp").reindex(grid)
    df["coin"] = coin_name
    n_raw = df["price"].notna().sum()
    # fill only short gaps (<= 3h); longer gaps stay missing rather than
    # becoming flat forward-filled prices that hide (or fake) depegs
    df["price"] = df["price"].interpolate(method="time", limit=3, limit_area="inside")
    n_interpolate = df["price"].notna().sum()
    df = df.reset_index().rename(columns={"index": "timestamp"})
    n_grid = len(df)
    df = df[["coin", "timestamp", "price"]].dropna(subset=["price"])

    print(f"  -> SUCCESS {coin_name}: {len(df):,} hourly rows | {df['timestamp'].min():%Y-%m-%d %H:%M} .. {df['timestamp'].max():%Y-%m-%d %H:%M} "
          f"| raw coverage {n_raw / n_grid:.1%}, interpolated {n_interpolate/n_grid:.1%}, {n_grid - len(df):,} hrs left missing")
    return df


def fetch_all_coin_prices() -> pd.DataFrame:
    """Fetch price data for all coins from DefiLlama."""

    print("=" * 70)
    print("Fetching hourly price data for all coins")
    print("=" * 70)

    price_frames = []
    failed_coins = []
    end_date = DEFAULT_END
    start_date = START_DATE if START_DATE else "2018-01-01"

    for coin_name, spec in DEFILLAMA_FIAT_COIN_SPECS.items():
        if coin_name not in FIAT_COINS:
            continue

        coin_end = spec.get("end", end_date)
        coin_start = spec.get("start", start_date)

        df_coin = fetch_defillama_hourly(coin_name, spec["id"], coin_start, coin_end)

        if not df_coin.empty:
            price_frames.append(df_coin)
        else:
            failed_coins.append(coin_name)

    if not price_frames:
        raise RuntimeError("No price data collected for any coin!")

    all_prices = pd.concat(price_frames, ignore_index=True)
    all_prices = all_prices.sort_values(["coin", "timestamp"]).reset_index(drop=True)
    # fastparquet mis-writes datetime64[s] (round-trips to 1970); store as ns
    all_prices["timestamp"] = all_prices["timestamp"].astype("datetime64[ns, UTC]")

    if failed_coins:
        print(f"\nWARNING: Failed to fetch data for: {failed_coins}")

    return all_prices


def fetch_binance_month(symbol: str, month: str) -> pd.DataFrame | None:
    """A 404 means the pair genuinely didn't trade that month (delisted /
    not yet listed) -- but data.binance.vision also intermittently 404s or
    drops the connection on a fine month, so retry before trusting a 404."""
    url = (f"https://data.binance.vision/data/spot/monthly/klines/{symbol}/1h/"
           f"{symbol}-1h-{month}.zip")
    last_status = None
    r = None
    for attempt in range(4):
        try:
            r = SESSION.get(url, timeout=60)
            last_status = r.status_code
            if r.status_code == 200:
                break
            if r.status_code != 404:
                r.raise_for_status()
        except requests.RequestException:
            pass
        time.sleep(2.0 * (attempt + 1))
    else:
        if last_status == 404:
            return None
        raise RuntimeError(f"Binance {symbol} {month}: repeated failures (last status {last_status})")
    if last_status == 404:
        return None
    with zipfile.ZipFile(io.BytesIO(r.content)) as z:
        raw = z.read(z.namelist()[0]).decode("utf-8")
    df = pd.read_csv(io.StringIO(raw), header=None, names=BINANCE_KLINE_COLS)
    # Binance switched kline archives from millisecond to microsecond open_time
    # from Jan 2025 onward (13-digit vs 16-digit). Parsing 2025+ as ms silently
    # overflows to the year 57385, which then gets dropped by the date-range
    # filter downstream with no error -- detect the unit from the magnitude.
    unit = "us" if df["open_time"].iloc[0] > 10**14 else "ms"
    df["hour"] = pd.to_datetime(df["open_time"], unit=unit, utc=True).dt.floor("h")
    for c in ["open", "high", "low", "close", "volume", "quote_volume", "taker_buy_base_volume"]:
        df[c] = df[c].astype(float)
    df["trades"] = df["trades"].astype(int)
    return df[["hour", "open", "high", "low", "close", "volume", "quote_volume",
               "trades", "taker_buy_base_volume"]]


def fetch_binance_range(symbol: str, start_s: str, end_s: str) -> pd.DataFrame:
    # data.binance.vision intermittently 404s a month that demonstrably exists
    # (confirmed by re-requesting it seconds later) -- back-to-back retries
    # inside fetch_binance_month don't reliably clear this, but a second pass
    # after a longer pause does. So collect misses and retry them once at the
    # end rather than trusting a 404 immediately.
    #
    # NOTE: a run of missing months that survives the retry (e.g. 2022-10
    # through 2023-02 for USDCUSDT/USDPUSDT/TUSDUSDT) isn't a fetch failure
    # here -- Binance suspended spot trading on USDC/USDP/TUSD pairs on
    # 2022-09-29 (auto-converting balances to BUSD) and reinstated them
    # around March 2023. Those months genuinely have zero trades, and
    # fill_hourly_volume_gaps (below) fills them with 0 accordingly.
    months = list(pd.period_range(start_s, end_s, freq="M").strftime("%Y-%m"))
    results = {}
    for m in months:
        results[m] = fetch_binance_month(symbol, m)
        time.sleep(0.3)

    missing = [m for m, d in results.items() if d is None]
    if missing:
        print(f"  {symbol}: {len(missing)} month(s) came back empty on first pass "
              f"({missing[:3]}{'...' if len(missing) > 3 else ''}), retrying after a pause ...", flush=True)
        time.sleep(10)
        for m in missing:
            results[m] = fetch_binance_month(symbol, m)
            time.sleep(1.0)
        still_missing = [m for m, d in results.items() if d is None]
        if still_missing:
            print(f"  {symbol}: confirmed missing (real gap, not flakiness): {still_missing}", flush=True)

    parts = [d for d in results.values() if d is not None]
    if not parts:
        return pd.DataFrame()
    df = pd.concat(parts, ignore_index=True)
    df = df[(df["hour"] >= pd.Timestamp(start_s, tz="UTC")) &
            (df["hour"] <= pd.Timestamp(end_s, tz="UTC"))]
    return df.sort_values("hour").drop_duplicates("hour")


def fetch_bitfinex_range(symbol: str, start_s: str, end_s: str) -> pd.DataFrame:
    start_ms = int(pd.Timestamp(start_s, tz="UTC").timestamp() * 1000)
    end_ms = int(pd.Timestamp(end_s, tz="UTC").timestamp() * 1000)
    rows = []
    cur = start_ms
    while cur < end_ms:
        r = None
        for attempt in range(5):
            try:
                r = SESSION.get(f"https://api-pub.bitfinex.com/v2/candles/trade:1h:{symbol}/hist",
                                 params={"start": cur, "end": end_ms, "limit": 10000, "sort": 1},
                                 timeout=30)
                if r.status_code == 200:
                    break
            except requests.RequestException:
                pass
            time.sleep(3.0 * (attempt + 1))  # Bitfinex rate-limits aggressively / drops connections
        else:
            raise RuntimeError(f"Bitfinex {symbol}: repeated failures at ts={cur}")
        batch = r.json()
        if not batch:
            break
        rows.extend(batch)
        cur = batch[-1][0] + 3600_000
        time.sleep(2.0)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows, columns=["ms", "open", "close", "high", "low", "volume"])
    df["hour"] = pd.to_datetime(df["ms"], unit="ms", utc=True).dt.floor("h")
    return df[["hour", "volume"]].sort_values("hour").drop_duplicates("hour")


def fill_hourly_volume_gaps(df: pd.DataFrame, start_s: str, end_s: str) -> pd.DataFrame:
    """Reindex a single exchange source to a complete hourly grid over its
    configured coverage window and fill missing hours with zero volume.

    A thinly-traded pair's candle endpoint returns no candle at all for an
    hour with zero trades -- confirmed by comparing coverage density across
    coins on the same venue: DAI's Bitfinex book (tDAIUSD) has real candles
    for only ~68% of its hourly window, vs ~100% for the far more liquid
    USDT/USD book (tUSTUSD) on the same exchange. The same logic covers a
    full trading *halt*, not just thin trading -- e.g. Binance suspended
    USDC/USDP/TUSD spot pairs entirely from 2022-09-29 to ~2023-03, which
    shows up as ~5 months of missing archives per fetch_binance_range. Either
    way it's a genuine zero, not a data gap, so it should be filled with 0 --
    otherwise a join against price treats "no candle" as "no data" and either
    drops the hour (inner join) or leaves it NaN (left join) when it should
    read as an actual quiet/halted hour.
    """
    if df.empty:
        return df
    grid = pd.date_range(pd.Timestamp(start_s, tz="UTC").floor("h"),
                          pd.Timestamp(end_s, tz="UTC"), freq="h")
    df = df.set_index("hour").reindex(grid)
    df = df.fillna(0.0)
    df = df.reset_index().rename(columns={"index": "hour"})
    return df


def main():
    """Main execution - fetches all data."""
    t0 = time.time()

    print("=" * 70)
    print("STABLECOIN DEPEG DATA FETCHER")
    print(f"Study window: {START_DATE} to {END_DATE}")
    print(f"Coins: {len(FIAT_COINS)} fiat-backed coins")
    print("=" * 70)

    # Fetch price data
    price_df = fetch_all_coin_prices()
    print("\nPrice data summary:")
    print(price_df.groupby("coin")["timestamp"].agg(["min", "max", "count"]))

    # Fetch volume data from exchanges (Binance, Bitfinex, Coinbase)
    print("\nFetching exchange volume data...")
    frames = []
    for coin, sources in EXCHANGE_OHLCV_SPECS.items():
        coin_parts = []
        for src in sources:
            end_s = src.get("end", END_DATE)
            print(f"Fetching {coin} <- {src['venue']}:{src['symbol']} {src['start']} -> {end_s} ...", flush=True)
            if src["venue"] == "binance":
                df = fetch_binance_range(src["symbol"], src["start"], end_s)
            elif src["venue"] == "bitfinex":
                df = fetch_bitfinex_range(src["symbol"], src["start"], end_s)
            else:
                raise ValueError(f"Unknown venue {src['venue']}")
            if df.empty:
                print(f"  WARNING: no data for {coin} from {src['venue']}:{src['symbol']}")
                continue
            df = fill_hourly_volume_gaps(df, src["start"], end_s)
            df["venue"] = src["venue"]
            df["quote_currency"] = src["quote"]
            coin_parts.append(df)
            print(f"  -> {len(df):,} hourly rows | {df['hour'].min():%Y-%m-%d %H:%M} .. {df['hour'].max():%Y-%m-%d %H:%M}")
        if not coin_parts:
            print(f"  ERROR: no exchange data collected for {coin} at all")
            continue
        # sources are listed primary-first; drop_duplicates(keep="first") after
        # sorting is stable, so an earlier (primary) source's row always wins
        # over a later (fallback) source's row for the same hour
        coin_df = pd.concat(coin_parts, ignore_index=True).sort_values("hour", kind="stable").drop_duplicates("hour")
        # Volume only: Binance rows carry OHLC alongside volume, but price
        # comes exclusively from DefiLlama, so drop any OHLC columns here
        # rather than letting a second, uncoordinated price source into the
        # output. quote_currency/quote_volume also dropped per earlier
        # decision -- not needed since price no longer comes from these pairs.
        coin_df = coin_df.drop(columns=[c for c in ["open", "high", "low", "close", "quote_currency", "quote_volume"] if c in coin_df.columns])
        coin_df.insert(0, "coin", coin)
        frames.append(coin_df)

    # NOTE: this final aggregation/save was previously indented *inside* the
    # loop above, so it ran once per coin (rewriting the parquet from scratch
    # every time, and making the "no data at all" check below unreachable).
    # It now runs exactly once, after every coin has been attempted.
    if not frames:
        raise RuntimeError("No exchange OHLCV collected for any coin!")

    volume_df = pd.concat(frames, ignore_index=True)
    # Join price (DefiLlama, the near-complete reference series) with volume
    # (exchanges). Now that each source's genuinely-zero-trade hours are
    # filled with 0 rather than missing entirely (see fill_hourly_volume_gaps),
    # a left join on price is the right choice: any hour still missing
    # exchange data after that fill is a real "outside this source's tracked
    # coverage window" case (e.g. after BUSD's Binance pair was delisted),
    # and should stay NaN rather than being dropped, as an inner join would.
    price_for_merge = price_df.rename(columns={"timestamp": "hour"})
    out = pd.merge(price_for_merge, volume_df, on=["coin", "hour"], how="left", indicator=True)
    out = out.sort_values(["coin", "hour"]).reset_index(drop=True)

    # Mark venue transitions per coin so any rolling volume-change window
    # (1h/6h/24h, etc.) computed downstream can avoid spanning a source
    # switch -- a raw volume jump right at a switch reflects a change in
    # data source, not a change in the coin's actual trading activity.
    # Computed only over rows that actually have exchange coverage (venue
    # notna): with a left join, a row can now legitimately have no exchange
    # data at all (outside every source's window), and treating that as a
    # "venue switch" would be comparing two absences, not a real transition.
    # `venue_switch` flags the first row of each new venue segment (True on
    # the first covered row of a coin's series too, since that's the start of
    # its first segment). `hours_since_venue_start` is the elapsed calendar
    # hours since that segment began (1 at the switch row itself), so a
    # change calculation can require e.g. hours_since_venue_start >= window + 1
    # before trusting it, and set the change to NaN otherwise. Both are NaN
    # on rows with no exchange coverage, since the concept doesn't apply there.
    covered = out.loc[out["venue"].notna(), ["coin", "hour", "venue"]].sort_values(["coin", "hour"]).copy()
    covered["venue_switch"] = covered.groupby("coin")["venue"].transform(lambda s: s != s.shift(1))
    segment_id = covered.groupby("coin")["venue_switch"].cumsum()
    segment_start_hour = covered.groupby(["coin", segment_id])["hour"].transform("min")
    covered["hours_since_venue_start"] = (
        (covered["hour"] - segment_start_hour).dt.total_seconds() / 3600.0
    ).astype(int) + 1
    out = out.merge(covered[["coin", "hour", "venue_switch", "hours_since_venue_start"]],
                     on=["coin", "hour"], how="left")

    matched = int((out["_merge"] == "both").sum())
    price_only = int((out["_merge"] == "left_only").sum())
    out = out.drop(columns="_merge")
    volume_unmatched = len(volume_df) - matched
    print(f"  Left join on price: {matched:,} rows have both price and exchange volume, "
          f"{price_only:,} price-only hours have no exchange coverage "
          f"({volume_unmatched:,} exchange rows fell outside price's coverage and were dropped)")

    for c in ["volume", "trades", "taker_buy_base_volume"]:
        if c not in out.columns:
            out[c] = pd.NA
    out = out[["coin", "hour", "price", "volume", "trades", "taker_buy_base_volume",
               "venue", "venue_switch", "hours_since_venue_start"]]
    out.to_parquet(OUT_PATH, index=False)
    print(f"\nSaved {len(out):,} rows -> {OUT_PATH}")
    print(out.groupby("coin").agg(start=("hour", "min"), end=("hour", "max"),
                                    rows=("hour", "size")).to_string())
    print(f"\nTotal runtime: {time.time() - t0:,.1f}s")


if __name__ == "__main__":
    main()