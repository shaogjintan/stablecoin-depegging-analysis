"""
Produce the "Data Feasibility" numbers/table for the midterm slide:
  - which coins overlap between price and on-chain data, and their common date range
  - timestamp frequency and missing-hour counts (the real coverage check)
  - number of independent depeg episodes (NOT raw transaction counts)
  - number of positive vs negative prediction windows per horizon, and how
    many of each SURVIVE after dropping embargoed/straddling hours
"""
import pandas as pd
from config import PROCESSED_DATA_DIR, REPORT_DIR, FORECAST_HORIZONS_HOURS, EMBARGO_HOURS


def intersection_table(price: pd.DataFrame, onchain: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for coin in sorted(set(price["coin"]) | set(onchain["coin"])):
        p = price[price["coin"] == coin]
        o = onchain[onchain["coin"] == coin]
        has_price = not p.empty
        has_onchain = not o.empty
        common_start = max(p["hour"].min() if has_price else pd.NaT,
                            o["hour"].min() if has_onchain else pd.NaT)
        common_end = min(p["hour"].max() if has_price else pd.NaT,
                          o["hour"].max() if has_onchain else pd.NaT)
        expected_hours = None
        if pd.notna(common_start) and pd.notna(common_end):
            expected_hours = int((common_end - common_start) / pd.Timedelta(hours=1)) + 1
        rows.append({
            "coin": coin,
            "has_price_data": has_price,
            "has_onchain_data": has_onchain,
            "price_rows": len(p),
            "onchain_rows": len(o),
            "common_start": common_start,
            "common_end": common_end,
            "expected_hours_in_range": expected_hours,
        })
    return pd.DataFrame(rows)


def missing_data_check(master: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for coin, g in master.groupby("coin"):
        n_total = len(g)
        n_missing_price = g["close"].isna().sum()
        n_missing_onchain = g["tx_count"].isna().sum()
        rows.append({
            "coin": coin,
            "n_hours": n_total,
            "missing_price_hours": int(n_missing_price),
            "missing_price_pct": round(100 * n_missing_price / n_total, 1) if n_total else None,
            "missing_onchain_hours": int(n_missing_onchain),
            "missing_onchain_pct": round(100 * n_missing_onchain / n_total, 1) if n_total else None,
        })
    return pd.DataFrame(rows)


def episode_summary(episodes: pd.DataFrame) -> pd.DataFrame:
    if episodes.empty:
        return pd.DataFrame(columns=["coin", "n_episodes", "total_depeg_hours"])
    return (
        episodes.groupby("coin")
        .agg(n_episodes=("episode_id", "nunique"), total_depeg_hours=("n_hours", "sum"))
        .reset_index()
    )


def window_balance_table(master: pd.DataFrame) -> pd.DataFrame:
    """For each horizon: total usable (non-NaN) windows, and the positive/negative split. 
    """
    rows = []
    for h in FORECAST_HORIZONS_HOURS:
        col = f"label_h{h}"
        usable = master[master[col].notna()]
        n_pos = int((usable[col] == 1).sum())
        n_neg = int((usable[col] == 0).sum())
        rows.append({
            "horizon_hours": h,
            "usable_windows": n_pos + n_neg,
            "positive_windows": n_pos,
            "negative_windows": n_neg,
            "positive_rate_pct": round(100 * n_pos / (n_pos + n_neg), 2) if (n_pos + n_neg) else None,
            "excluded_active_episode_hours": int(master["in_episode"].sum()),
            "excluded_right_censored_hours": int(master[col].isna().sum() - master["in_episode"].sum()),
        })
    return pd.DataFrame(rows)


def main():
    price = pd.read_parquet(PROCESSED_DATA_DIR / "hourly_price.parquet")
    price = price.rename(columns={"timestamp": "hour"})
    onchain = pd.read_parquet(PROCESSED_DATA_DIR / "hourly_onchain_features.parquet")
    master = pd.read_parquet(PROCESSED_DATA_DIR / "master_hourly_dataset.parquet")
    episodes = pd.read_parquet(PROCESSED_DATA_DIR / "depeg_episodes.parquet")

    tbl1 = intersection_table(price, onchain)
    tbl2 = missing_data_check(master)
    tbl3 = episode_summary(episodes)
    tbl4 = window_balance_table(master)

    tbl1.to_csv(REPORT_DIR / "1_coin_intersection.csv", index=False)
    tbl2.to_csv(REPORT_DIR / "2_missing_data_check.csv", index=False)
    tbl3.to_csv(REPORT_DIR / "3_episode_summary.csv", index=False)
    tbl4.to_csv(REPORT_DIR / "4_window_balance.csv", index=False)
    episodes.to_csv(REPORT_DIR / "5_episode_detail.csv", index=False)

    print("=== 1. Coin intersection ===\n", tbl1, "\n")
    print("=== 2. Missing data check ===\n", tbl2, "\n")
    print("=== 3. Independent depeg episodes (the real sample size) ===\n", tbl3, "\n")
    print("=== 4. Positive vs negative prediction windows per horizon ===\n", tbl4, "\n")
    print(f"(Embargo = {EMBARGO_HOURS}h purged around each episode boundary to prevent leakage.)")


if __name__ == "__main__":
    main()