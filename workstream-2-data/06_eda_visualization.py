"""
Exploratory visualization for the midterm slide deck. This is diagnostic only.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import skew
from scipy.spatial import ConvexHull
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import LabelEncoder
import umap

from config import PROCESSED_DATA_DIR, REPORT_DIR, EMBARGO_HOURS

FEATURE_COLS = [
    "price_dev", "tx_count", "total_volume", "active_senders",
    #"total_volume_usd", "whale_tx_count", "median_transfer_usd", "transfer_gini",
]

LOG_TRANSFORM_COLS = ["tx_count", "total_volume", "active_senders",] #"total_volume_usd", "whale_tx_count", ]

COIN_MARKERS = ["o", "s", "^", "D", "v", "P", "X"]


def tag_regime(df: pd.DataFrame, episodes: pd.DataFrame, pre_window_hours: int = 24) -> pd.DataFrame:
    df = df.copy()
    df["regime"] = "normal"
    for _, ep in episodes.iterrows():
        coin_mask = df["coin"] == ep["coin"]
        pre_mask = coin_mask & (df["hour"] >= ep["start"] - pd.Timedelta(hours=pre_window_hours)) & (df["hour"] < ep["start"])
        in_mask = coin_mask & (df["hour"] >= ep["start"]) & (df["hour"] <= ep["end"])
        post_mask = coin_mask & (df["hour"] > ep["end"]) & (df["hour"] <= ep["end"] + pd.Timedelta(hours=pre_window_hours))
        df.loc[pre_mask, "regime"] = "pre_episode"
        df.loc[in_mask, "regime"] = "in_episode"
        df.loc[post_mask, "regime"] = "post_episode"
    return df


def plot_episode_timelines(df: pd.DataFrame, episodes: pd.DataFrame):
    coins = sorted(df["coin"].unique())
    fig, axes = plt.subplots(len(coins), 2, figsize=(12, 3 * len(coins)), sharex=False)
    if len(coins) == 1:
        axes = axes.reshape(1, 2)
    for i, coin in enumerate(coins):
        g = df[df["coin"] == coin].sort_values("hour")
        is_stablecoin = g["is_stablecoin"].iloc[0] if "is_stablecoin" in g.columns and len(g) else True
        axes[i, 0].plot(g["hour"], g["close"])
        if is_stablecoin:
            axes[i, 0].axhline(1.0, color="grey", linestyle="--", linewidth=0.8)
            axes[i, 0].set_title(f"{coin}: price")
        else:
            axes[i, 0].set_title(f"{coin}: price (not a stablecoin -- no $1 peg to reference)")
        axes[i, 1].plot(g["hour"], g["tx_count"], color="tab:orange")
        axes[i, 1].set_title(f"{coin}: hourly tx count")
        for _, ep in episodes[episodes["coin"] == coin].iterrows():
            axes[i, 0].axvspan(ep["start"], ep["end"], color="red", alpha=0.2)
            axes[i, 1].axvspan(ep["start"], ep["end"], color="red", alpha=0.2)
    fig.tight_layout()
    out = REPORT_DIR / "episode_timelines.png"
    fig.savefig(out, dpi=150)
    print(f"Saved {out}")


def plot_embeddings(df: pd.DataFrame):
    data = df.dropna(subset=FEATURE_COLS).copy()
    if len(data) < 20:
        print("Not enough complete rows for an embedding plot -- skipping.")
        return
    if "WLUNA" in data["coin"].unique():
        print("WARNING: WLUNA rows present in plot_embeddings() input -- these should have "
              "been filtered out by the caller (WLUNA has no depeg episodes, so it would "
              "silently pollute the 'normal' regime cluster). Continuing, but treat any "
              "output below with caution.")

    print("Skewness before/after log1p transform:")
    for c in FEATURE_COLS:
        raw_skew = skew(data[c].dropna())
        if c in LOG_TRANSFORM_COLS:
            transformed_skew = skew(np.log1p(data[c].clip(lower=0)).dropna())
            print(f"  {c:22s} raw={raw_skew:7.2f}   log1p={transformed_skew:7.2f}  (transformed)")
        else:
            print(f"  {c:22s} raw={raw_skew:7.2f}   (left as-is)")

    feature_matrix = data[FEATURE_COLS].copy()
    for c in LOG_TRANSFORM_COLS:
        feature_matrix[c] = np.log1p(feature_matrix[c].clip(lower=0))

    X = StandardScaler().fit_transform(feature_matrix.to_numpy())

    pca = PCA(n_components=2)
    emb_pca = pca.fit_transform(X)
    explained = pca.explained_variance_ratio_

    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
    emb_umap = reducer.fit_transform(X)

    tsne = TSNE(n_components=2, perplexity=min(30, max(5, len(data) // 10)), random_state=42)
    emb_tsne = tsne.fit_transform(X)

    regimes = data["regime"].to_numpy()
    coins = data["coin"].to_numpy()
    unique_coins = sorted(data["coin"].unique())
    coin_marker = {c: COIN_MARKERS[i % len(COIN_MARKERS)] for i, c in enumerate(unique_coins)}
    colors = {"normal": "lightgrey", "pre_episode": "orange", "in_episode": "red", "post_episode": "purple"}

    fig, axes = plt.subplots(1, 3, figsize=(19, 5))

    embeddings = [
        (emb_pca, "PCA: hourly feature vectors by regime"),
        (emb_umap, "UMAP: hourly feature vectors by regime"),
        (emb_tsne, "t-SNE: hourly feature vectors by regime"),
    ]

    def draw_hull(ax, points, color):
        """Convex hull outline for a regime's points -- makes the EXTENT of
        each regime's spread visible at a glance, not just point color. Needs
        >=3 non-collinear points; silently skipped otherwise (common for
        small regimes like in_episode with a single short crash)."""
        if len(points) < 3:
            return
        try:
            hull = ConvexHull(points)
            for simplex in hull.simplices:
                ax.plot(points[simplex, 0], points[simplex, 1], color=color, alpha=0.4, linewidth=1)
            ax.fill(points[hull.vertices, 0], points[hull.vertices, 1], color=color, alpha=0.08)
        except Exception:
            pass  # degenerate (collinear) point sets -- skip rather than crash the whole plot

    for ax, (embedding, title) in zip(axes, embeddings):
        for regime, c in colors.items():
            mask = regimes == regime
            draw_hull(ax, embedding[mask], c)
            for coin in unique_coins:
                coin_mask = mask & (coins == coin)
                if coin_mask.sum() == 0:
                    continue
                ax.scatter(embedding[coin_mask, 0], embedding[coin_mask, 1], s=14, c=c,
                           marker=coin_marker[coin], alpha=0.75,
                           label=f"{regime} / {coin}" if ax is axes[0] else None)
        ax.set_title(title)

    axes[0].set_xlabel(f"PC1 ({explained[0] * 100:.1f}% variance)")
    axes[0].set_ylabel(f"PC2 ({explained[1] * 100:.1f}% variance)")

    regime_handles = [plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=c, markersize=8, label=r)
                       for r, c in colors.items()]
    coin_handles = [plt.Line2D([0], [0], marker=m, color="black", linestyle="", markersize=8, label=coin)
                     for coin, m in coin_marker.items()]
    fig.legend(handles=regime_handles, loc="upper center", bbox_to_anchor=(0.15, -0.02), ncol=4, fontsize=8, title="regime (color)")
    fig.legend(handles=coin_handles, loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=len(unique_coins), fontsize=8, title="coin (marker)")

    fig.tight_layout()
    out = REPORT_DIR / "regime_embeddings.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved {out}")
    print(
        f"PCA variance explained: "
        f"PC1={explained[0]:.3f}, "
        f"PC2={explained[1]:.3f}, "
        f"total={explained[:2].sum():.3f}"
    )
    loadings = pd.DataFrame(
        pca.components_.T,
        index=FEATURE_COLS,
        columns=["PC1", "PC2"]
    )
    print(loadings)
    print("\nRegime counts:")
    print(data["regime"].value_counts())
    print("\nRegime proportions:")
    print(data["regime"].value_counts(normalize=True).mul(100).round(2))

    print("\nUMAP regime centroids:")

    for regime in colors:
        mask = regimes == regime
        if mask.sum() == 0:
            print(f"{regime:12s}: (no rows -- e.g. a right-censored episode with no observed recovery)")
            continue
        centroid = emb_umap[mask].mean(axis=0)
        print(
            f"{regime:12s}: "
            f"({centroid[0]:8.3f}, {centroid[1]:8.3f}), "
            f"n={mask.sum()}"
        )

    print("\nt-SNE regime centroids:")

    for regime in colors:
        mask = regimes == regime
        if mask.sum() == 0:
            print(f"{regime:12s}: (no rows)")
            continue
        centroid = emb_tsne[mask].mean(axis=0)
        print(
            f"{regime:12s}: "
            f"({centroid[0]:8.3f}, {centroid[1]:8.3f}), "
            f"n={mask.sum()}"
        )

    le = LabelEncoder()
    regime_labels = le.fit_transform(regimes)

    umap_silhouette = silhouette_score(
        emb_umap,
        regime_labels
    )

    tsne_silhouette = silhouette_score(
        emb_tsne,
        regime_labels
    )

    original_silhouette = silhouette_score(
        X,
        regime_labels
    )

    print("\nSilhouette scores:")
    print(f"Original feature space: {original_silhouette:.3f}")
    print(f"UMAP embedding:         {umap_silhouette:.3f}")
    print(f"t-SNE embedding:        {tsne_silhouette:.3f}")

    subset_mask = np.isin(
        regimes,
        ["normal", "pre_episode"]
    )

    X_subset = X[subset_mask]
    regime_subset = regimes[subset_mask]

    le = LabelEncoder()
    labels_subset = le.fit_transform(regime_subset)

    print(
        "\nNormal vs pre-episode silhouette "
        f"(original space): "
        f"{silhouette_score(X_subset, labels_subset):.3f}"
    )

    umap_subset = emb_umap[subset_mask]

    print(
        "Normal vs pre-episode silhouette "
        f"(UMAP): "
        f"{silhouette_score(umap_subset, labels_subset):.3f}"
    )

    print("\nFeature means by regime:")
    print(
        data.groupby("regime")[FEATURE_COLS]
            .mean()
            .round(3)
    )
    print("\nFeature medians by regime:")
    print(
        data.groupby("regime")[FEATURE_COLS]
            .median()
            .round(3)
    )

def plot_coverage_heatmap(master: pd.DataFrame):
    coins = sorted(master["coin"].unique())
    hours = sorted(master["hour"].unique())
    hour_idx = {h: i for i, h in enumerate(hours)}

    grid = np.zeros((len(coins), len(hours)))  # 0=neither, 1=price only, 2=onchain only, 3=both
    for i, coin in enumerate(coins):
        g = master[master["coin"] == coin]
        has_price = g["close"].notna()
        has_onchain = g["tx_count"].notna()
        for j, h, hp, ho in zip(g["hour"].map(hour_idx), g["hour"], has_price, has_onchain):
            grid[i, j] = (1 if hp else 0) + (2 if ho else 0)

    fig, ax = plt.subplots(figsize=(14, 0.6 * len(coins) + 1.5))
    im = ax.imshow(grid, aspect="auto", cmap="RdYlGn", vmin=0, vmax=3, interpolation="nearest")
    ax.set_yticks(range(len(coins)))
    ax.set_yticklabels(coins)
    tick_step = max(1, len(hours) // 15)
    ax.set_xticks(range(0, len(hours), tick_step))
    ax.set_xticklabels([pd.Timestamp(hours[i]).strftime("%m-%d") for i in range(0, len(hours), tick_step)],
                        rotation=45, ha="right")
    ax.set_title("Data coverage by coin-hour (green=both price+on-chain present, red=neither)")
    cbar = fig.colorbar(im, ax=ax, ticks=[0, 1, 2, 3])
    cbar.ax.set_yticklabels(["neither", "price only", "on-chain only", "both"])
    fig.tight_layout()
    out = REPORT_DIR / "coverage_heatmap.png"
    fig.savefig(out, dpi=150)
    print(f"Saved {out}")


def main():
    master = pd.read_parquet(PROCESSED_DATA_DIR / "master_hourly_dataset.parquet")
    episodes = pd.read_parquet(PROCESSED_DATA_DIR / "depeg_episodes.parquet")

    plot_coverage_heatmap(master)

    all_coins_with_regime = tag_regime(master.copy(), episodes)
    plot_episode_timelines(all_coins_with_regime, episodes)

    stablecoins_only = all_coins_with_regime[all_coins_with_regime["is_stablecoin"]].copy()
    plot_embeddings(stablecoins_only)


if __name__ == "__main__":
    main()