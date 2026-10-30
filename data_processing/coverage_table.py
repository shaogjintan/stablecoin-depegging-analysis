
"""
Generate separate market and on-chain coverage tables.

All inputs are read from master.parquet (committed), which already contains
the market columns (trading_volume), the on-chain columns (tx_count) and the
common_sample flag identified by merge_and_label.py.

  - Market coverage uses every coin-hour with observed trading_volume
    (NOT restricted to the common sample).
  - On-chain coverage is restricted to coin-hours where common_sample == True.

Outputs:
  - market_coverage_table.csv
  - onchain_coverage_table.csv
"""

import pandas as pd
from config import PROCESSED_DATA_DIR, FIAT_COINS

def calculate_coverage(df,source_name,observed_col,restrict_to_common):
    """
    Calculate coverage for one data source.

    Start date, end date, and observed hours are based on coin-hours where
    observed_col is not missing. If restrict_to_common is True, only
    coin-hours where common_sample == True are counted.
    """
    mask = df[observed_col].notna()
    if restrict_to_common:
        # Restrict source data to common coin-hours
        mask &= df["common_sample"].fillna(False).astype(bool)
    observed = df.loc[mask,["coin", "hour"]].drop_duplicates()
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
    label = "COMMON-SAMPLE COVERAGE" if restrict_to_common else "FULL-SAMPLE COVERAGE"
    print(f"\n{source_name.upper()} {label}")
    print(coverage.to_string(index=False))
    return coverage

def main():
    # LOAD DATASET 
    master = pd.read_parquet(
        PROCESSED_DATA_DIR / "master.parquet",
        columns=["coin", "hour", "common_sample", "trading_volume", "tx_count"],
    )
    master["hour"] = pd.to_datetime(master["hour"], utc=True)

    # VALIDATE COMMON SAMPLE
    if "common_sample" not in master.columns:
        raise ValueError(
            "master.parquet does not contain common_sample. "
            "Run merge_and_label.py first.")

    common_hours = (
        master.loc[master["common_sample"]]
        .groupby("coin")["hour"]
        .nunique()
        .to_dict())

    # TABLEs 
    market_coverage = calculate_coverage(
        df=master,
        source_name="Market",
        observed_col="trading_volume",
        restrict_to_common=False,)
    onchain_coverage = calculate_coverage(
        df=master,
        source_name="On-chain",
        observed_col="tx_count",
        restrict_to_common=True,)

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
