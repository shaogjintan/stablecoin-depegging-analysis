
"""
Generate separate market and on-chain coverage tables.

Both tables include the number of common coin-hours
identified by 03_merge_and_label.py.

Outputs:
  - market_coverage_table.csv
  - onchain_coverage_table.csv
"""

import pandas as pd
from config import PROCESSED_DATA_DIR, FIAT_COINS

def calculate_coverage(df,source_name,observed_col,master):
    """
    Calculate coverage restricted to the common sample.

    Start date, end date, and observed hours are based
    on coin-hours where common_sample == True.
    """
    common = master.loc[master["common_sample"],["coin", "hour"]].drop_duplicates()
    # Restrict source data to common coin-hours
    observed = df.loc[df[observed_col].notna(),["coin", "hour"]].drop_duplicates()
    observed = observed.merge(common,on=["coin", "hour"],how="inner",validate="one_to_one")
    rows = []
    for coin in sorted(FIAT_COINS):
        g = observed.loc[observed["coin"] == coin]
        if g.empty:
            rows.append({
                "coin": coin,
                "start_date": pd.NaT,
                "end_date": pd.NaT,
                "observed_hours": 0,
                "expected_hours": 0,
                "coverage_pct": 0.0,
            })
            continue

        start = g["hour"].min()
        end = g["hour"].max()
        expected_hours = int((end - start).total_seconds() / 3600) + 1
        observed_hours = g["hour"].nunique()
        rows.append({
            "coin": coin,
            "start_date": start,
            "end_date": end,
            "observed_hours": observed_hours,
            "expected_hours": expected_hours,
            "coverage_pct": (
                100 * observed_hours / expected_hours
            ),
        })
    coverage = pd.DataFrame(rows)
    print(f"\n{source_name.upper()} COMMON-SAMPLE COVERAGE")
    print(coverage.to_string(index=False))
    return coverage

def main():
    # LOAD DATASETS
    market = pd.read_parquet(PROCESSED_DATA_DIR / "hourly_stablecoin_market_features.parquet")
    onchain = pd.read_parquet(PROCESSED_DATA_DIR / "hourly_stablecoin_onchain_features.parquet")
    master = pd.read_parquet(PROCESSED_DATA_DIR / "master.parquet")

    for df in [market, onchain, master]:
        df["hour"] = pd.to_datetime(df["hour"], utc=True)

    # VALIDATE COMMON SAMPLE
    if "common_sample" not in master.columns:
        raise ValueError(
            "master.parquet does not contain common_sample. "
            "Run 03_merge_and_label.py first.")

    common_hours = (
        master.loc[master["common_sample"]]
        .groupby("coin")["hour"]
        .nunique()
        .to_dict())

    # TABLEs 
    market_coverage = calculate_coverage(
        df=market,
        source_name="Market",
        observed_col="trading_volume",
        master=master,)
    onchain_coverage = calculate_coverage(
        df=onchain,
        source_name="On-chain",
        observed_col="tx_count",
        master=master,)

    # EXPORT TABLES
    market_coverage.to_csv(
        PROCESSED_DATA_DIR / "market_coverage_table.csv",
        index=False)

    onchain_coverage.to_csv(
        PROCESSED_DATA_DIR / "onchain_coverage_table.csv",
        index=False)
    print("\nBoth coverage tables saved successfully.")

if __name__ == "__main__":
    main()
