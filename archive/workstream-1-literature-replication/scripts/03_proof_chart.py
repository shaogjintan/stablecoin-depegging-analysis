#!/usr/bin/env python3
"""
TASK 3: "Daily Frequency Flaw" proof chart — UST intraday crash cascade.
Hourly vs daily detection of the May 2022 UST depeg.
Outputs: outputs/proof_daily_vs_hourly.png (300 dpi) and .pdf
"""
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

DARK_RED = "#8B0000"
THRESH = 0.99


def main():
    df = pd.read_csv("data/may_2022_hourly_prices.csv", parse_dates=["timestamp"])
    ust = df[df["coin"] == "UST"].set_index("timestamp").sort_index()

    # study window: May 7 00:00 .. May 11 00:00 UTC
    start, end = pd.Timestamp("2022-05-07", tz="UTC"), pd.Timestamp("2022-05-11", tz="UTC")
    w = ust[(ust.index >= start) & (ust.index <= end)]["price"]

    # FIRST hour below $0.99 (hourly detection)
    below = w[w < THRESH]
    assert len(below) > 0, "no sub-0.99 hour found in window"
    t_hourly = below.index.min()

    # FIRST full UTC day whose daily close is below $0.99; daily detection
    # happens at that day's close -> marker at the last hour of that day.
    daily_close = w.resample("D").last()
    daily_below = daily_close[daily_close < THRESH]
    assert len(daily_below) > 0, "no sub-0.99 daily close found in window"
    day_daily = daily_below.index.min()
    t_daily = day_daily + pd.Timedelta(hours=23)  # end of the first such day

    lost_hours = round((t_daily - t_hourly).total_seconds() / 3600, 1)

    fig, ax = plt.subplots(figsize=(11, 6), dpi=300)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    ax.plot(w.index, w.values, color=DARK_RED, linewidth=2, label="UST hourly price")

    ax.axhline(THRESH, color="gray", linestyle="--", linewidth=1.4)
    ax.axhline(1.01, color="gray", linestyle="--", linewidth=1.4, alpha=0.7)
    ax.text(w.index[2], THRESH + 0.0012, "Depeg Threshold ($0.99)", color="gray",
            fontsize=9, va="bottom")

    # shading between the two detection points
    ax.axvspan(t_hourly, t_daily, color="#FFF3B0", alpha=0.6, zorder=0)

    ax.scatter([t_hourly], [w.loc[t_hourly]], marker="*", s=420, color="orange",
               edgecolor="black", zorder=6, label="Hourly detection (Our approach)")
    ax.scatter([t_daily], [w.asof(t_daily)], marker="D", s=90, color="black",
               zorder=6, label="Daily close detection (Lee et al. 2025)")

    ax.annotate(f"Warning window lost by daily models: ~{lost_hours:.0f} hours",
                xy=(t_hourly + (t_daily - t_hourly) / 2, max(w.values) * 0.93),
                ha="center", fontsize=11, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.4", facecolor="#FFFDF0",
                          edgecolor="#C9A227"))

    ax.set_title("Why Daily Models Fail: UST Intraday Crash Cascade (May 7-10, 2022)",
                 fontsize=14, fontweight="bold", pad=12)
    ax.set_ylabel("Price (USD)")
    ax.set_ylim(min(w.values) * 0.985, 1.02)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=12))
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=8)
    ax.legend(loc="lower left", fontsize=9, framealpha=0.95)
    ax.grid(True, alpha=0.25, linewidth=0.6)

    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    fig.tight_layout()
    fig.savefig("outputs/proof_daily_vs_hourly.png", dpi=300)
    fig.savefig("outputs/proof_daily_vs_hourly.pdf")
    print(f"hourly detection : {t_hourly}  (UST = {w.loc[t_hourly]:.4f})")
    print(f"daily detection  : {t_daily}  (UST = {w.asof(t_daily):.4f})")
    print(f"lost warning window: {lost_hours:.1f} hours")
    print("saved outputs/proof_daily_vs_hourly.png (300dpi) and .pdf")


if __name__ == "__main__":
    main()
