"""
Random Forest feature importance for the fixed RF config (tag rf_500_d8_l50).

Refits the same forest as the canonical run (500 trees, max_depth 8,
min_samples_leaf 50, max_features sqrt, class_weight balanced_subsample,
random_state 42) on each expanding-window fold's full training set (years < Y),
with train-only median imputation as in train.py, and records sklearn's
impurity importance. Answers: which features does RF rely on, and how much
weight do the on-chain features get once they are available (Tier 2)?

Importance = mean decrease in impurity, normalised to sum to 1 per fold. It
describes what the forest uses, not what improves out-of-sample accuracy.

A pooled test PR-AUC is also written as a reproduction check against
reports/modelling_comparison.csv (rf_500_d8_l50).

OUTPUT (reports/rf_importance/)
  per_fold.csv        importance per horizon x tier x test year x feature
  summary.csv         mean / min / max across the 5 folds
  pr_auc_check.csv    pooled test PR-AUC of the refit forests
  rf_importance_t2.png  Tier 2, h=6 and h=24, coloured market vs on-chain
  rf_importance_tier_compare.png  market-feature shares, Tier 1 vs Tier 2
  rf_importance_slide.png  presentation version: top features + grouped bars

Usage (from the repo root; the refits take ~15 minutes):
    python modelling/reporting/rf_importance.py
    python modelling/reporting/rf_importance.py --plot-only   # redraw from summary.csv
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import average_precision_score  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO_ROOT), str(REPO_ROOT / "data_processing")]

from modelling.models.base import MedianImputer  # noqa: E402
from modelling.models.random_forest_model import RFModel  # noqa: E402
from modelling.prepare_data import MARKET_FEATURES, load_folds  # noqa: E402

FIXED_PARAMS = dict(n_estimators=500, max_depth=8, min_samples_leaf=50,
                    max_features="sqrt", class_weight="balanced_subsample",
                    random_state=42)
HORIZONS, TIERS = (6, 24), (1, 2)

# market vs on-chain (validated categorical pair); text stays in ink colours
MARKET_COLOUR, ONCHAIN_COLOUR = "#2a78d6", "#eb6834"
TIER_COLOURS = {1: "#6250d6", 2: "#1baf7a"}
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e4e3de"


def fit_importances(out_dir: Path) -> pd.DataFrame:
    rows, check = [], []
    for h in HORIZONS:
        for tier in TIERS:
            t0 = time.time()
            ys, ps = [], []
            for f in load_folds(horizon=h, tier=tier):
                imp = MedianImputer().fit(f["X_train"])
                model = RFModel(**FIXED_PARAMS).fit(imp.transform(f["X_train"]), f["y_train"])
                ps.append(pd.Series(model.predict_proba(imp.transform(f["X_test"]))[:, 1]))
                ys.append(f["y_test"].reset_index(drop=True))
                rows += [{"horizon": h, "tier": tier, "test_year": f["year"],
                          "feature": feat, "importance": v}
                         for feat, v in zip(f["X_train"].columns, model.feature_importances_)]
            y, p = pd.concat(ys), pd.concat(ps)
            check.append({"horizon": h, "tier": tier, "n": len(y), "pos_rate": y.mean(),
                          "pr_auc": average_precision_score(y, p)})
            print(f"h={h} tier={tier}: pooled PR-AUC {check[-1]['pr_auc']:.4f} "
                  f"({time.time() - t0:.0f}s)", flush=True)

    per_fold = pd.DataFrame(rows)
    per_fold.to_csv(out_dir / "per_fold.csv", index=False)
    pd.DataFrame(check).to_csv(out_dir / "pr_auc_check.csv", index=False)
    summary = (per_fold.groupby(["horizon", "tier", "feature"])["importance"]
               .agg(["mean", "min", "max"]).reset_index())
    summary["group"] = summary["feature"].map(
        lambda c: "market" if c in MARKET_FEATURES else "onchain")
    summary = summary.sort_values(["horizon", "tier", "mean"], ascending=[True, True, False])
    summary.to_csv(out_dir / "summary.csv", index=False)
    return summary


def pretty(feature: str) -> str:
    """'tx_count_logdiff_24h' -> 'Tx count 24h', 'log_return_6h' -> 'Log return 6h'."""
    name = feature.replace("_logdiff", "").replace("pct_change", "% chg").replace("_", " ")
    return name[:1].upper() + name[1:]


def plot(summary: pd.DataFrame, out_dir: Path) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(13, 7.2), sharex=True)
    for ax, h in zip(axes, HORIZONS):
        d = summary[(summary["tier"] == 2) & (summary["horizon"] == h)].sort_values("mean")
        scale = 1 / d["mean"].sum()
        y = range(len(d))
        colours = [MARKET_COLOUR if g == "market" else ONCHAIN_COLOUR for g in d["group"]]
        ax.barh(y, d["mean"] * scale, color=colours, height=0.68, zorder=2)
        ax.hlines(y, d["min"] * scale, d["max"] * scale, color=INK, lw=1, alpha=0.55, zorder=3)
        ax.set_yticks(list(y), [pretty(f) for f in d["feature"]], color=INK)
        ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
        ax.grid(axis="x", color=GRID, lw=0.8, zorder=0)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(GRID)
        ax.tick_params(axis="y", length=0)
        ax.tick_params(axis="x", colors=MUTED)
        onchain_share = d.loc[d["group"] == "onchain", "mean"].sum() * scale
        ax.set_title(f"h = {h}h  ·  on-chain total: {onchain_share:.0%}", loc="left",
                     fontsize=12, color=INK, fontweight="bold")
        ax.set_xlabel("Share of importance (bar = mean of 5 folds, line = fold range)",
                      color=MUTED, fontsize=9)

    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in (MARKET_COLOUR, ONCHAIN_COLOUR)]
    fig.legend(handles, ["Market (Tier 1)", "On-chain (added in Tier 2)"],
               loc="upper right", ncol=2, frameon=False, fontsize=10)
    fig.suptitle("Random Forest feature importance, Tier 2", x=0.01, ha="left",
                 fontsize=15, fontweight="bold", color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.95), w_pad=3)
    path = out_dir / "rf_importance_t2.png"
    fig.savefig(path, dpi=200, facecolor="white")
    plt.close(fig)
    return path


def plot_slide(summary: pd.DataFrame, out_dir: Path, top: int = 4) -> Path:
    """Presentation version (Tier 2): top market features, the rest of the market
    features and all on-chain features as grouped bars, h=6 and h=24 side by side."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 6), sharex=True)
    for ax, h in zip(axes, HORIZONS):
        d = summary[(summary["tier"] == 2) & (summary["horizon"] == h)].copy()
        d["share"] = d["mean"] / d["mean"].sum()
        market = d[d["group"] == "market"].sort_values("share", ascending=False)
        onchain = d[d["group"] == "onchain"]
        rows = [(pretty(f), s, MARKET_COLOUR) for f, s in zip(market["feature"][:top],
                                                              market["share"][:top])]
        rows.append((f"Other {len(market) - top} market, combined", market["share"][top:].sum(),
                     MARKET_COLOUR))
        rows.append((f"All {len(onchain)} on-chain, combined", onchain["share"].sum(),
                     ONCHAIN_COLOUR))
        rows = rows[::-1]  # barh draws bottom-up
        y = range(len(rows))
        ax.barh(y, [r[1] for r in rows], color=[r[2] for r in rows], height=0.62, zorder=2)
        for yi, (_, share, _) in zip(y, rows):
            ax.text(share + 0.004, yi, f"{share:.0%}", va="center", fontsize=13, color=INK)
        ax.set_yticks(list(y), [r[0] for r in rows], fontsize=13, color=INK)
        ax.get_yticklabels()[0].set_fontweight("bold")
        ax.set_xlim(0, 0.25)
        ax.set_xticks([0, 0.05, 0.10, 0.15, 0.20, 0.25])
        ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
        ax.grid(axis="x", color=GRID, lw=0.8, zorder=0)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(GRID)
        ax.tick_params(axis="y", length=0)
        ax.tick_params(axis="x", colors=MUTED, labelsize=11)
        ax.set_xlabel("Share of importance", fontsize=12, color=MUTED)
        ax.set_title(f"h = {h}h", loc="left", fontsize=15, color=INK, fontweight="bold")

    each = summary[(summary["tier"] == 2) & (summary["group"] == "onchain")].copy()
    each = each["mean"] / each["horizon"].map(
        summary[summary["tier"] == 2].groupby("horizon")["mean"].sum())
    fig.text(0.5, 0.075, f"All 12 on-chain features together ≈ one top market feature; "
             f"each on its own gets only {each.min():.0%}–{each.max():.0%}",
             ha="center", fontsize=12, color=MUTED)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in (MARKET_COLOUR, ONCHAIN_COLOUR)]
    fig.legend(handles, ["Market features", "On-chain features"], loc="lower center",
               ncol=2, frameon=False, fontsize=12)
    fig.suptitle("Random Forest feature importance (Tier 2: market + on-chain)",
                 fontsize=17, fontweight="bold", color=INK)
    fig.tight_layout(rect=(0, 0.1, 1, 0.97), w_pad=4)
    path = out_dir / "rf_importance_slide.png"
    fig.savefig(path, dpi=220, facecolor="white")
    plt.close(fig)
    return path


def plot_tier_compare(summary: pd.DataFrame, out_dir: Path) -> Path:
    """Share of each market feature in Tier 1 vs Tier 2: where the on-chain weight comes from."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 6.4), sharex=True)
    for ax, h in zip(axes, HORIZONS):
        d = summary[summary["horizon"] == h].copy()
        d["share"] = d["mean"] / d.groupby("tier")["mean"].transform("sum")
        wide = d[d["group"] == "market"].pivot(index="feature", columns="tier", values="share")
        wide = wide.sort_values(1)
        y = pd.Series(range(len(wide)), dtype=float)
        ax.barh(y + 0.19, wide[1], height=0.36, color=TIER_COLOURS[1], zorder=2)
        ax.barh(y - 0.19, wide[2], height=0.36, color=TIER_COLOURS[2], zorder=2)
        ax.set_yticks(list(y), [pretty(f) for f in wide.index], color=INK)
        ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
        ax.grid(axis="x", color=GRID, lw=0.8, zorder=0)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(GRID)
        ax.tick_params(axis="y", length=0)
        ax.tick_params(axis="x", colors=MUTED)
        onchain_share = 1 - wide[2].sum()
        ax.set_title(f"h = {h}h  ·  on-chain takes {onchain_share:.0%} in Tier 2", loc="left",
                     fontsize=12, color=INK, fontweight="bold")
        ax.set_xlabel("Share of importance within the tier (mean of 5 folds)",
                      color=MUTED, fontsize=9)

    handles = [plt.Rectangle((0, 0), 1, 1, color=TIER_COLOURS[t]) for t in TIERS]
    fig.legend(handles, ["Tier 1 (market only)", "Tier 2 (market + on-chain)"],
               loc="upper right", ncol=2, frameon=False, fontsize=10)
    fig.suptitle("Random Forest: market-feature importance, Tier 1 vs Tier 2", x=0.01,
                 ha="left", fontsize=15, fontweight="bold", color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.95), w_pad=3)
    path = out_dir / "rf_importance_tier_compare.png"
    fig.savefig(path, dpi=200, facecolor="white")
    plt.close(fig)
    return path


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", default=str(REPO_ROOT / "reports/rf_importance"))
    parser.add_argument("--plot-only", action="store_true",
                        help="skip the refits and redraw from summary.csv")
    args = parser.parse_args(argv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = (pd.read_csv(out_dir / "summary.csv") if args.plot_only
               else fit_importances(out_dir))
    print(f"saved {plot(summary, out_dir)}")
    print(f"saved {plot_tier_compare(summary, out_dir)}")
    print(f"saved {plot_slide(summary, out_dir)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
