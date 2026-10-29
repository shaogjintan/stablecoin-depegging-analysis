"""
SHAP interpretability for the tuned XGBoost folds (Phase 5).

Loads the per-fold models saved by modelling/train.py
(results/models/{model}_t{tier}_h{horizon}_{year}.joblib), re-loads the
corresponding folds via modelling/prepare_data.py, and computes TreeExplainer
SHAP values on each fold's out-of-sample test rows, pooled across folds.

Outputs (under reports/):
  shap_ranking_{tag}.csv + .png      mean |SHAP| ranking, top-15 bar plot
  beeswarm_{tag}.png                 beeswarm summary (subsampled for speed)
  dependence_{tag}_{feature}.png     dependence plots for the top-3 features
  group_share_{tag}.csv + .png       market vs on-chain |SHAP| share (tier 2
                                     only) -- the key research-question evidence

Usage (from the repo root):
    python modelling/feature_importance/shap_analysis.py --model xgboost --horizon 24 --tier 2
    python modelling/feature_importance/shap_analysis.py --model xgboost --horizon 24 --tier 1
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from modelling.prepare_data import (  # noqa: E402
    MARKET_FEATURES,
    ONCHAIN_FEATURES,
    load_folds,
)

REPORTS_DIR = REPO_ROOT / "reports"
FIG_DIR = REPORTS_DIR / "figures"
RESULTS_MODELS = REPO_ROOT / "results" / "models"

BINARY_GROUPS = {"market": MARKET_FEATURES, "onchain": ONCHAIN_FEATURES}


def collect_shap(model: str, tier: int, horizon: int):
    """Pooled SHAP values (n_total, p), the pooled feature matrix, and names."""
    folds = load_folds(horizon=horizon, tier=tier)
    shap_parts, X_parts = [], []
    for fold in folds:
        year = fold["year"]
        artifact = RESULTS_MODELS / f"{model}_t{tier}_h{horizon}_{year}.joblib"
        if not artifact.exists():
            raise FileNotFoundError(
                f"{artifact} not found -- run modelling/train.py for this "
                f"config first (e.g. --model {model} --horizon {horizon} "
                f"--tier {tier} --tune)")
        m = joblib.load(artifact)
        X = fold["X_test"].reset_index(drop=True)
        explainer = shap.TreeExplainer(m.clf_)
        sv = np.asarray(explainer.shap_values(X), dtype=float)
        if sv.ndim == 3:  # older shap returned (n, p, 2) for binary models
            sv = sv[:, :, 1]
        shap_parts.append(sv)
        X_parts.append(X)
        print(f"  fold {year}: SHAP on {len(X):,} OOS test rows")
    SV = np.vstack(shap_parts)
    Xp = pd.concat(X_parts, ignore_index=True)
    features = list(folds[0]["features"])
    assert SV.shape == Xp.shape, (SV.shape, Xp.shape)
    return SV, Xp, features


def ranking(SV, Xp, features, tag: str, out_dir: Path) -> pd.DataFrame:
    mean_abs = np.abs(SV).mean(axis=0)
    order = np.argsort(-mean_abs)
    df = pd.DataFrame({
        "feature": [features[i] for i in order],
        "mean_abs_shap": mean_abs[order],
        "share_of_total_abs": mean_abs[order] / mean_abs.sum(),
    })
    df.to_csv(out_dir / f"shap_ranking_{tag}.csv", index=False)

    top = df.head(15).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.barh(top["feature"], top["mean_abs_shap"], color="#1f77b4")
    ax.set_xlabel("mean |SHAP value| (pooled OOS)")
    ax.set_title(f"SHAP feature importance — {tag}")
    fig.tight_layout()
    fig.savefig(out_dir / f"shap_ranking_{tag}.png", dpi=150)
    plt.close(fig)
    return df


def beeswarm(SV, Xp, features, tag: str, out_dir: Path,
             max_rows: int = 15000, seed: int = 42) -> None:
    if len(SV) > max_rows:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(SV), size=max_rows, replace=False)
        SV, Xp = SV[idx], Xp.iloc[idx]
    shap.summary_plot(SV, Xp, feature_names=features, show=False, max_display=20)
    fig = plt.gcf()
    fig.set_size_inches(8, 7)
    fig.savefig(out_dir / f"beeswarm_{tag}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def dependence_top3(rank_df: pd.DataFrame, SV, Xp, features, tag: str,
                    out_dir: Path) -> None:
    for feat in rank_df["feature"].head(3):
        shap.dependence_plot(feat, SV, Xp, feature_names=features, show=False)
        fig = plt.gcf()
        fig.set_size_inches(7, 5)
        safe = re.sub(r"[^A-Za-z0-9_.-]", "_", feat)
        fig.savefig(out_dir / f"dependence_{tag}_{safe}.png", dpi=150,
                    bbox_inches="tight")
        plt.close(fig)


def group_share(SV, features, tag: str, out_dir: Path) -> pd.DataFrame:
    """Share of pooled |SHAP| attributable to market vs on-chain features."""
    abs_sv = np.abs(SV)
    total = abs_sv.sum()
    rows = {}
    for group, feats in BINARY_GROUPS.items():
        idx = [features.index(c) for c in feats if c in features]
        rows[group] = float(abs_sv[:, idx].sum() / total)
    df = pd.DataFrame([rows])
    df.to_csv(out_dir / f"group_share_{tag}.csv", index=False)

    fig, ax = plt.subplots(figsize=(6, 2.2))
    left = 0.0
    for group, color in (("market", "#4C72B0"), ("onchain", "#DD8452")):
        ax.barh([0], [df[group].iloc[0]], left=left, color=color,
                label=f"{group} ({df[group].iloc[0]:.1%})")
        ax.text(left + df[group].iloc[0] / 2, 0, f"{df[group].iloc[0]:.1%}",
                ha="center", va="center", color="white", fontweight="bold")
        left += df[group].iloc[0]
    ax.set_xlim(0, 1)
    ax.set_yticks([])
    ax.set_xlabel("share of pooled |SHAP|")
    ax.set_title(f"Market vs on-chain contribution — {tag}")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.35), ncol=2)
    fig.tight_layout()
    fig.savefig(out_dir / f"group_share_{tag}.png", dpi=150)
    plt.close(fig)
    return df


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="xgboost")
    parser.add_argument("--horizon", type=int, choices=[1, 6, 24], default=24)
    parser.add_argument("--tier", type=int, choices=[1, 2], default=2)
    parser.add_argument("--tag", default=None)
    args = parser.parse_args(argv)

    tag = args.tag or f"{args.model}_t{args.tier}_h{args.horizon}"
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    print(f"SHAP analysis: model={args.model} horizon={args.horizon} "
          f"tier={args.tier} tag={tag}")

    SV, Xp, features = collect_shap(args.model, args.tier, args.horizon)
    print(f"pooled SHAP matrix: {SV.shape}")

    rank_df = ranking(SV, Xp, features, tag, FIG_DIR)
    print("\nTop-10 mean |SHAP|:")
    print(rank_df.head(10).round(4).to_string(index=False))

    beeswarm(SV, Xp, features, tag, FIG_DIR)
    dependence_top3(rank_df, SV, Xp, features, tag, FIG_DIR)

    if args.tier == 2:
        gs = group_share(SV, features, tag, FIG_DIR)
        print("\nMarket vs on-chain share of |SHAP|:")
        print(gs.round(4).to_string(index=False))

    print(f"\nOutputs written to {FIG_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
