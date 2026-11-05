"""LOCAL-ONLY driver: regenerate the deck's coverage table (P5) from
master.parquet with an explicit definition.

Definition used: for each coin, expected hours = its observed span in
master.parquet (first to last hour); observed hours = rows where every
Tier-2 feature (market volume + the four on-chain raw columns) is present.
Coverage % = observed / expected. This replaces the stale P5 numbers, which
reproduce in no file in the repo.

Usage: python plans/coverage_table.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "data_processing"))

from config import FIAT_COINS, PROCESSED_DATA_DIR  # noqa: E402

TIER2_COLS = ["trading_volume", "tx_count", "tx_volume", "active_senders", "active_receivers"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(REPO / "reports/coverage_table.csv"))
    args = ap.parse_args()

    m = pd.read_parquet(PROCESSED_DATA_DIR / "master.parquet",
                        columns=["coin", "hour"] + TIER2_COLS)
    m["hour"] = pd.to_datetime(m["hour"], utc=True)
    rows = []
    for coin in FIAT_COINS:
        g = m[m["coin"] == coin]
        span = (g["hour"].max() - g["hour"].min()) / pd.Timedelta(hours=1) + 1
        full = g.dropna(subset=TIER2_COLS)
        rows.append({
            "coin": coin,
            "start_date": g["hour"].min().date(),
            "end_date": g["hour"].max().date(),
            "expected_hours": int(span),
            "observed_hours": int(len(full)),
            "coverage_pct": round(100.0 * len(full) / span, 1),
        })
    out = pd.DataFrame(rows)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    print(out.to_string(index=False))
    print(f"\nSaved {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
