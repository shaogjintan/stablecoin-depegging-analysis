"""
07_depeg_by_slide_definition.py

NEW, STANDALONE script -- does not modify 04_merge_and_label.py or any
other teammate's file. It reads the already-built master_hourly_dataset.parquet
(produced by 04) and applies a SECOND, INDEPENDENT depeg definition on top
of it, so we can see exactly how much it changes the headline numbers.

Why this exists:
  - 04_merge_and_label.py's flag_episodes() flags a SYMMETRIC deviation:
        abs(close - 1.0) > DEPEG_BAND
    i.e. a depeg if price is EITHER above 1+band OR below 1-band.
  - Slide 5 defines a depeg as a DOWNWARD deviation only:
        close <= 1.0 - DEPEG_BAND
    ("a downward deviation of at least 1% below the peg")
  This script uses the slide's downward-only definition, with the same
  magnitude (DEPEG_BAND) and persistence (DEPEG_MIN_DURATION_HOURS) from
  config.py -- only the direction differs from 04.

OUTPUTS (all under reports/):
  - 07_depeg_comparison_table.csv          (old symmetric vs new downward-only, per coin)
  - 07_depeg_episodes_slide_definition.csv (episode-level detail, slide definition)
  - 07_price_timeline_with_depegs.png      (price over time, red dots = slide-def depeg hours)
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from config import PROCESSED_DATA_DIR, REPORT_DIR, DEPEG_BAND, DEPEG_MIN_DURATION_HOURS

FOCUS_COINS = ["USDT", "USDC", "DAI", "UST"]


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


def build_comparison_table(df: pd.DataFrame) -> pd.DataFrame:
    """Old (04's symmetric code definition) vs new (slide's downward-only definition),
    per focus coin: episode count and total depeg hours under each."""
    rows = []
    for coin in FOCUS_COINS:
        g = df[df["coin"] == coin]
        rows.append({
            "coin": coin,
            "episodes_code_symmetric": g["episode_id"].dropna().nunique(),
            "hours_code_symmetric": int(g["in_episode"].sum()),
            "episodes_slide_downward_only": g["slide_episode_id"].dropna().nunique(),
            "hours_slide_downward_only": int(g["in_slide_episode"].sum()),
        })
    return pd.DataFrame(rows)


def build_episode_table(df: pd.DataFrame) -> pd.DataFrame:
    eps = (
        df[df["in_slide_episode"]]
        .groupby(["coin", "slide_episode_id"])
        .agg(start=("hour", "min"), end=("hour", "max"), n_hours=("hour", "count"),
             min_price=("close", "min"), max_price=("close", "max"))
        .reset_index()
        .sort_values("start")
    )
    return eps


def plot_timeline_with_depegs(df: pd.DataFrame) -> None:
    # sharex=True so all 4 panels span the SAME date range -- lets you see
    # directly whether depegs across coins line up in time (contagion) or not.
    fig, axes = plt.subplots(len(FOCUS_COINS), 1, figsize=(14, 3 * len(FOCUS_COINS)), sharex=True)
    global_min, global_max = df["hour"].min(), df["hour"].max()
    for ax, coin in zip(axes, FOCUS_COINS):
        g = df[df["coin"] == coin].sort_values("hour")
        ax.plot(g["hour"], g["close"], linewidth=0.8, color="steelblue", label="price")
        ax.axhline(1.0, color="grey", linestyle="--", linewidth=0.8)
        depeg_rows = g[g["in_slide_episode"]]
        ax.scatter(depeg_rows["hour"], depeg_rows["close"], color="red", s=10, zorder=5,
                   label="depeg hour (slide definition: downward only)")
        ax.set_title(f"{coin}: price with slide-definition depeg episodes marked")
        ax.legend(fontsize=8, loc="upper right")
    axes[0].set_xlim(global_min, global_max)
    fig.tight_layout()
    out = REPORT_DIR / "07_price_timeline_with_depegs.png"
    fig.savefig(out, dpi=150)
    print(f"Saved {out}")


def main():
    master = pd.read_parquet(PROCESSED_DATA_DIR / "master_hourly_dataset.parquet")
    df = master[master["coin"].isin(FOCUS_COINS)].copy()

    df = flag_downward_episodes(df)

    print("=" * 70)
    print("DEPEG DEFINITION COMPARISON")
    print("  code (04_merge_and_label.py): symmetric, abs(price - 1) > band")
    print("  slides (slide 5):             downward only, price <= 1 - band")
    print("=" * 70)
    comparison = build_comparison_table(df)
    print(comparison.to_string(index=False))
    comparison.to_csv(REPORT_DIR / "07_depeg_comparison_table.csv", index=False)
    print(f"\nSaved -> {REPORT_DIR / '07_depeg_comparison_table.csv'}")

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
    simple.to_csv(REPORT_DIR / "07_depeg_summary_simple.csv", index=False)
    print(f"\nSaved -> {REPORT_DIR / '07_depeg_summary_simple.csv'}")

    episodes = build_episode_table(df)
    episodes.to_csv(REPORT_DIR / "07_depeg_episodes_slide_definition.csv", index=False)
    print(f"Saved -> {REPORT_DIR / '07_depeg_episodes_slide_definition.csv'} "
          f"({len(episodes)} episodes total across {FOCUS_COINS})")

    plot_timeline_with_depegs(df)


if __name__ == "__main__":
    main()
