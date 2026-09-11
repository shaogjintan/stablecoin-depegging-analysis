"""
09_threshold_sensitivity.py

Robustness/sensitivity check for the depeg magnitude threshold: how much
does the number of episodes (and total depeg hours) change per coin as we
move the band away from the chosen 1%?

Held FIXED: downward-only direction (slide 5's definition), persistence
filter (>= DEPEG_MIN_DURATION_HOURS consecutive hours, from config.py).
Held VARIABLE: the magnitude band itself, tested at 0.8%, 0.9%, 1.0%
(baseline), 1.2%, 1.5%.

NEW, STANDALONE script -- does not modify 04, 06, 07, or 08.

OUTPUTS:
  - reports/09_threshold_sensitivity_episodes.csv  (coin x threshold, episode counts)
  - reports/09_threshold_sensitivity_hours.csv      (coin x threshold, depeg hours)
"""
import numpy as np
import pandas as pd

from config import PROCESSED_DATA_DIR, REPORT_DIR, DEPEG_MIN_DURATION_HOURS

THRESHOLDS = [0.008, 0.009, 0.010, 0.012, 0.015]  # 0.8%, 0.9%, 1.0%, 1.2%, 1.5%


def compute_episode_stats(df: pd.DataFrame, band: float) -> pd.DataFrame:
    """Per-coin episode count + total depeg hours for a given magnitude band,
    downward-only, persistence held fixed at DEPEG_MIN_DURATION_HOURS."""
    rows = []
    for coin, g in df.groupby("coin"):
        g = g.sort_values("hour")
        below = (g["price_dev"] < -band).fillna(False).to_numpy()
        run_change = np.r_[True, below[1:] != below[:-1]]
        run_id = pd.Series(np.cumsum(run_change), index=g.index)
        below_s = pd.Series(below, index=g.index)

        run_lengths = run_id[below_s].value_counts()
        valid = run_lengths[run_lengths >= DEPEG_MIN_DURATION_HOURS]
        rows.append({"coin": coin, "n_episodes": len(valid), "n_hours": int(valid.sum())})
    return pd.DataFrame(rows).set_index("coin")


def main():
    master = pd.read_parquet(PROCESSED_DATA_DIR / "master_hourly_dataset.parquet")
    df = master[master["is_stablecoin"]].copy()
    coins = sorted(df["coin"].unique())

    episode_cols, hour_cols = {}, {}
    for band in THRESHOLDS:
        stats = compute_episode_stats(df, band)
        label = f"{band * 100:.1f}%"
        episode_cols[label] = stats["n_episodes"]
        hour_cols[label] = stats["n_hours"]

    episodes_table = pd.DataFrame(episode_cols).reindex(coins)
    hours_table = pd.DataFrame(hour_cols).reindex(coins)

    print("=" * 70)
    print("THRESHOLD SENSITIVITY -- number of episodes per coin, by band")
    print("(persistence held fixed at >=", DEPEG_MIN_DURATION_HOURS, "consecutive hours; downward-only)")
    print("=" * 70)
    print(episodes_table.to_string())
    episodes_table.to_csv(REPORT_DIR / "09_threshold_sensitivity_episodes.csv")
    print(f"\nSaved -> {REPORT_DIR / '09_threshold_sensitivity_episodes.csv'}")

    print("\n" + "=" * 70)
    print("THRESHOLD SENSITIVITY -- total depeg hours per coin, by band")
    print("=" * 70)
    print(hours_table.to_string())
    hours_table.to_csv(REPORT_DIR / "09_threshold_sensitivity_hours.csv")
    print(f"\nSaved -> {REPORT_DIR / '09_threshold_sensitivity_hours.csv'}")


if __name__ == "__main__":
    main()
