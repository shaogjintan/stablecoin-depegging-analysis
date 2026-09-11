"""
08_depeg_comparison_all_coins.py

Same comparison as 07_depeg_by_slide_definition.py (code's symmetric
definition vs the slide's downward-only definition), but for EVERY
stablecoin you have data for, not just the 4 headline coins.

Auxiliary tokens (WLUNA, MKR, CRV) are excluded via the `is_stablecoin`
flag already computed in 04_merge_and_label.py -- they have no $1 peg to
depeg from.

NEW, STANDALONE script -- does not modify 04, 06, or 07.

OUTPUT:
  - reports/08_depeg_comparison_all_coins.csv
"""
import numpy as np
import pandas as pd

from config import PROCESSED_DATA_DIR, REPORT_DIR, DEPEG_BAND, DEPEG_MIN_DURATION_HOURS


def flag_downward_episodes(df: pd.DataFrame) -> pd.DataFrame:
    """Slide-5 definition: close <= 1 - DEPEG_BAND, persisting >= DEPEG_MIN_DURATION_HOURS
    consecutive hours. Adds `in_slide_episode` (bool) and `slide_episode_id` (str/None)."""
    df = df.copy()
    df["is_below_band"] = (df["price_dev"] < -DEPEG_BAND).fillna(False)

    frames = []
    for coin, g in df.groupby("coin"):
        g = g.sort_values("hour").copy()
        below = g["is_below_band"].to_numpy()
        run_change = np.r_[True, below[1:] != below[:-1]]
        run_id = np.cumsum(run_change)
        g["run_id"] = run_id

        run_lengths = g[g["is_below_band"]].groupby("run_id")["run_id"].transform("size")
        g["in_slide_episode"] = False
        g.loc[g["is_below_band"], "in_slide_episode"] = run_lengths >= DEPEG_MIN_DURATION_HOURS

        qualifying_runs = sorted(g.loc[g["in_slide_episode"], "run_id"].unique())
        run_to_num = {r: i + 1 for i, r in enumerate(qualifying_runs)}
        episode_num = g["run_id"].map(run_to_num)
        g["slide_episode_id"] = np.where(
            g["in_slide_episode"],
            coin + "_" + episode_num.astype("Int64").astype(str),
            None,
        )
        frames.append(g)

    return pd.concat(frames, ignore_index=True)


def build_comparison_table(df: pd.DataFrame, coins: list) -> pd.DataFrame:
    rows = []
    for coin in coins:
        g = df[df["coin"] == coin]
        rows.append({
            "coin": coin,
            "episodes_code_symmetric": g["episode_id"].dropna().nunique(),
            "hours_code_symmetric": int(g["in_episode"].sum()),
            "episodes_slide_downward_only": g["slide_episode_id"].dropna().nunique(),
            "hours_slide_downward_only": int(g["in_slide_episode"].sum()),
        })
    return pd.DataFrame(rows)


def main():
    master = pd.read_parquet(PROCESSED_DATA_DIR / "master_hourly_dataset.parquet")
    df = master[master["is_stablecoin"]].copy()

    coins = sorted(df["coin"].unique())
    df = flag_downward_episodes(df)

    print("=" * 70)
    print("DEPEG DEFINITION COMPARISON -- ALL STABLECOINS")
    print("  code (04_merge_and_label.py): symmetric, abs(price - 1) > band")
    print("  slides (slide 5):             downward only, price <= 1 - band")
    print("=" * 70)
    comparison = build_comparison_table(df, coins)
    comparison = comparison.sort_values("episodes_slide_downward_only", ascending=False)
    print(comparison.to_string(index=False))

    out_path = REPORT_DIR / "08_depeg_comparison_all_coins.csv"
    comparison.to_csv(out_path, index=False)
    print(f"\nSaved -> {out_path}")

    print("\n" + "=" * 70)
    print("SIMPLE SUMMARY (slide definition only: downward, >=1%, >=2h)")
    print("=" * 70)
    simple = comparison[["coin", "episodes_slide_downward_only", "hours_slide_downward_only"]].rename(
        columns={
            "episodes_slide_downward_only": "number_of_episodes",
            "hours_slide_downward_only": "number_of_depeg_hours",
        }
    )
    print(simple.to_string(index=False))
    simple_path = REPORT_DIR / "08_depeg_summary_simple.csv"
    simple.to_csv(simple_path, index=False)
    print(f"\nSaved -> {simple_path}")


if __name__ == "__main__":
    main()
