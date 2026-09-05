"""
Replication of the core result in:
    Lee, Y.-H., Chiu, Y.-F., & Hsieh, M.-H. (2025). Stablecoin depegging risk
    prediction. Pacific-Basin Finance Journal, 90, 102640.

WHAT WE ARE REPLICATING, EXACTLY
---------------------------------
Lee et al. is a full pipeline: 4 coins (USDT/USDC/BUSD/DAI) x 66 features
(price/volume, market-cap/supply, sentiment, volatility -- each block also
repeated for BTC and ETH as spillover regressors) x 3 models (LR/RF/XGBoost)
x SMOTE-tuned stratified split -> Table 5 (F1-led comparison).

Our workstream brief scopes this down to: "Run a simple Logistic Regression
and Random Forest using daily price/volatility indicators. Produce a table
showing their baseline accuracy/ROC-AUC." So this script replicates the
following specific, load-bearing pieces of their paper, and nothing else:

  (a) THE LABEL (Section 3.1, the Y equation).
      Their single biggest methodological contribution isn't a model, it's
      how they define "depegged" at all: a threshold that tightens for
      high-volume coins and loosens for low-volume ones, rather than one
      fixed band for every coin. We reproduce this formula exactly.

  (b) A REDUCED SLICE OF THEIR FEATURES (Table 1, blocks 1 and 4 --
      price/volume change rates and volatility -- for the coin itself,
      PLUS the same two blocks for BTC and ETH as spillover regressors).
      Their Table 1 has 4 feature blocks: (1) price/volume change rates,
      (2) market-cap/circulating-supply/total-supply change rates,
      (3) sentiment indices, (4) volatility indicators. We don't have
      market-cap/supply series or sentiment feeds for any asset, so blocks
      (2) and (3) are skipped entirely -- see "NOT replicating" below.
      One deliberate correction to their block (4): Price Deviation and
      Downward Price Deviation are both defined as RMS distance from "1"
      (the $1 peg). That's a meaningful stability metric for a stablecoin,
      but meaningless for BTC/ETH -- "distance from $1" for a $30,000 asset
      is just its price, squared, not a volatility signal. Lee et al.'s
      paper doesn't say they adjusted for this, but we do: BTC/ETH only
      contribute price/volume-change-rate and Realized Daily Volatility
      features (both scale-invariant), not peg-deviation ones.

  (c) THEIR MODELS + THEIR (FLAWED) SPLIT (Section 3.2, Table 5).
      Logistic Regression and Random Forest (we drop their third model,
      XGBoost, per the brief). Critically, we also copy their train/test
      split *as they did it*: stratified random sampling on the label,
      not a chronological split. This is deliberate, not an oversight --
      the whole point of "replicating" here is to reproduce their result
      faithfully first, flaws included, so a later script can swap in a
      chronological/purged split and show how much the numbers change.
      That contrast *is* the shortcomings slide.

WHAT WE ARE **NOT** TRYING TO REPLICATE
----------------------------------------
  - Market-cap/supply features and sentiment indices (Fear & Greed,
    Sentiment, Awareness) -- data we don't have for any asset. Sentiment
    turned out not to matter in their own results anyway (see their
    Table 6 -- no sentiment variable makes the top 15 predictive features).
  - XGBoost and their SMOTE hyperparameter search -- out of scope per brief.
  - Their exact coin roster, partially. They use USDT/USDC/BUSD/DAI; we run
    all four of those (BUSD included, via Yahoo Finance -- see DATA SOURCES
    below) plus pax, ustc, and wluna as a bonus -- the latter two are
    algorithmic coins Lee et al. excluded entirely, which happen to be the
    ones that actually failed. Seeing their method run on those two is
    itself a useful result (see the comment in main()).

DATA SOURCES
------------
All price (OHLC) and volume come from Yahoo Finance (via yfinance),
covering 2021-11-15 to 2023-12-31 -- a ~45-day lookback margin before
Lee et al.'s actual sample period (2022-01-01 to 2023-12-31) so the 30-day
rolling label/feature windows are already full on day 1 of the real
analysis window. Saved locally in data/ohlcv/*_ohlcv.csv (gitignored;
re-run the fetch script to regenerate).

We *don't* use ERC20-stablecoins/price_data (this project's other raw
price source) here, on purpose: those files have no volume column at all
(Lee et al.'s threshold formula needs trailing 30-day volume) and only
span Apr-Nov 2022, which would cut Lee et al.'s 2-year sample period down
to a 7-month one. CoinGecko's free API refuses to serve data this old
(>365 days) any more, which is why Yahoo Finance is being used instead of
the paper's actual source (CoinMarketCap) -- a data-access constraint,
not a design choice.
"""
import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix,
)

# stablecoin-depegging-analysis/models/lee_et_al_replication.py -> repo root
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OHLCV_DIR = os.path.join(REPO_ROOT, "data", "ohlcv")

# Our project's coin universe (matches ERC20-stablecoins/price_data' coins),
# PLUS busd so Lee et al.'s exact roster (usdt/usdc/busd/dai) is fully
# covered -- see docstring above for why ustc/wluna are a bonus, not a
# straight substitution.
COINS = ["usdt", "usdc", "dai", "pax", "busd", "ustc", "wluna"]
# Cross-asset spillover regressors, per Lee et al. Section 3.1: "predicting
# the depegging of USDT involves ... variables related to ... USDT, BTC,
# and ETH."
SPILLOVER_ASSETS = ["btc", "eth"]

# Lee et al.'s actual sample period -- we filter down to this window *after*
# computing rolling features, so the ~45-day lookback margin in the raw
# data is used only to warm up the 30-day windows, never scored.
ANALYSIS_START = pd.Timestamp("2022-01-01")
ANALYSIS_END = pd.Timestamp("2023-12-31")

# --- Carey (2023) / Kaiko dynamic-threshold constants, as used by Lee et al. ---
# ThreshD/U = 1 -/+ K / V_monthly^ALPHA. Lee et al. cite ALPHA = 1/3 as Carey's
# "optimal setting" and K = 10; we take both as given rather than re-deriving them.
ALPHA = 1 / 3
K = 10

RNG_SEED = 42     # fixed seed so the "faithful" stratified split is reproducible
TEST_SIZE = 0.3   # Lee et al. don't state their exact split ratio; 70/30 is standard


def load_ohlcv(sym: str) -> pd.DataFrame:
    """Load one asset's daily OHLCV from data/ohlcv/ (Yahoo Finance pull).

    Works identically for a stablecoin (usdt, usdc, ...) or a spillover
    asset (btc, eth) -- both are just daily OHLCV tables.
    """
    df = pd.read_csv(os.path.join(OHLCV_DIR, f"{sym}_ohlcv.csv"))
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").reset_index(drop=True)


def add_label(df: pd.DataFrame) -> pd.DataFrame:
    """REPLICATES: Lee et al. Section 3.1, the depegging label Y.

        ThreshD_t = 1 - K / V_monthly_t^ALPHA
        ThreshU_t = 1 + K / V_monthly_t^ALPHA
        Y_t = 1  if  P_low_t <= ThreshD_t  OR  P_high_t >= ThreshU_t
              0  otherwise

    where V_monthly_t is the trailing 30-day *sum* of daily trading volume
    (their "rolling windows sum of trading volume over a 30-day window").

    This is the mechanism that makes the threshold *dynamic*: a coin trading
    heavily this month gets a tight band (small K/V^(1/3)), a thinly-traded
    coin gets a loose one. It is also bilateral -- it fires on *either* an
    upward or downward peg break, which is Lee et al.'s own stated extension
    over Carey (2023)'s downward-only version.

    Note we use that day's own high/low against a threshold built from
    volume *up to and including* that day -- i.e. Y_t is a same-day label.
    The one-day-ahead *prediction* problem is set up separately in
    build_dataset() by shifting this label forward relative to the features.
    Only meaningful for a $1-pegged asset -- never call this on BTC/ETH.
    """
    df = df.copy()
    v_monthly = df["volume"].rolling(30, min_periods=30).sum()
    thresh_d = 1 - K / (v_monthly ** ALPHA)
    thresh_u = 1 + K / (v_monthly ** ALPHA)
    df["thresh_d"] = thresh_d
    df["thresh_u"] = thresh_u
    df["depeg"] = ((df["low"] <= thresh_d) | (df["high"] >= thresh_u)).astype(int)
    # First 29 days have no full 30-day volume window yet -> threshold is
    # undefined -> label is undefined (NaN), not "not depegged". Getting this
    # wrong would quietly manufacture 29 fake negative days per coin.
    df.loc[v_monthly.isna(), "depeg"] = np.nan
    return df


def rogers_satchell(row) -> float:
    """REPLICATES: Lee et al. Table 1, "Realized_Daily_Volatility".

    The Rogers & Satchell (1991) OHLC volatility estimator, as Lee et al.
    cite it via Grobys (2021):

        sigma_t = sqrt( ln(PH/PC)*ln(PH/PO) + ln(PL/PC)*ln(PL/PO) )

    Unlike a close-to-close return std, this uses a single day's own O/H/L/C
    to estimate that day's realized volatility -- no rolling window needed.
    It's scale-invariant (a ratio of prices), so unlike the peg-deviation
    features below, it's meaningful for BTC/ETH too, not just stablecoins.
    """
    po, ph, pl, pc = row["open"], row["high"], row["low"], row["close"]
    if min(po, ph, pl, pc) <= 0:
        return np.nan
    val = np.log(ph / pc) * np.log(ph / po) + np.log(pl / pc) * np.log(pl / po)
    # Floating point can push a near-zero value fractionally negative; clip
    # before sqrt rather than let it silently become NaN.
    return np.sqrt(max(val, 0.0))


def add_rate_of_change_features(df: pd.DataFrame, prefix: str = "") -> pd.DataFrame:
    """REPLICATES: Lee et al. Table 1, block (1) "trading price/volume
    change rate" -- plus Realized Daily Volatility from block (4), which
    (unlike the peg-deviation features) applies equally well to any asset.

    `prefix` namespaces the columns so the same function builds both the
    target coin's own features (prefix="") and a spillover asset's features
    (prefix="btc_" / "eth_") without name collisions once everything is
    merged into one row per day.

    Returns a `date` column plus the feature columns, so callers just
    merge this onto the main frame on `date`.
    """
    close = df["close"]
    vol = df["volume"]

    out = pd.DataFrame({"date": df["date"]})
    # Lee et al. use 1h/24h/7d/30d price % change and 24h/7d/30d volume %
    # change; we only have daily data, so 24h collapses to a plain 1-day
    # change and 1h has no daily analogue.
    out[f"{prefix}pct_change_1d"] = close.pct_change(1)
    out[f"{prefix}pct_change_7d"] = close.pct_change(7)
    out[f"{prefix}pct_change_30d"] = close.pct_change(30)
    out[f"{prefix}volume_pct_change_1d"] = vol.pct_change(1)
    out[f"{prefix}volume_pct_change_7d"] = vol.pct_change(7)
    out[f"{prefix}volume_pct_change_30d"] = vol.pct_change(30)
    out[f"{prefix}realized_daily_volatility"] = df.apply(rogers_satchell, axis=1)
    return out


def add_peg_deviation_features(df: pd.DataFrame, prefix: str = "") -> pd.DataFrame:
    """REPLICATES: Lee et al. Table 1, block (4) "Price_Deviation" and
    "Downward_Price_Deviation" (Kwon et al. 2023, as cited by Lee et al.):

        Price_Deviation_T    = sqrt( mean_{t-T..t}[ (P_close - 1)^2 ] )
        Downward_Deviation_T = sqrt( mean_{t-T..t}[ min(P_close - 1, 0)^2 ] )

    "1" here is the $1 peg -- these measure RMS distance from the peg
    (Price Deviation) vs. RMS distance *only counting downside misses*
    (Downward Deviation), each over a 5-day and a 30-day rolling window,
    exactly as Lee et al. set T. Only meaningful for a $1-pegged asset --
    call this on stablecoins only, never on BTC/ETH (see module docstring).
    """
    close = df["close"]
    out = pd.DataFrame({"date": df["date"]})

    dev = (close - 1) ** 2
    out[f"{prefix}price_deviation_5d"] = np.sqrt(dev.rolling(5).mean())
    out[f"{prefix}price_deviation_30d"] = np.sqrt(dev.rolling(30).mean())

    down_dev = close.apply(lambda p: min(p - 1, 0) ** 2)
    out[f"{prefix}downward_price_deviation_5d"] = np.sqrt(down_dev.rolling(5).mean())
    out[f"{prefix}downward_price_deviation_30d"] = np.sqrt(down_dev.rolling(30).mean())
    return out


def feature_columns() -> list[str]:
    """Full feature-column list used by every build_dataset() call: the
    coin's own rate-of-change + peg-deviation features, plus each
    spillover asset's rate-of-change features (no peg-deviation -- see
    module docstring for why).
    """
    own = [
        "pct_change_1d", "pct_change_7d", "pct_change_30d",
        "volume_pct_change_1d", "volume_pct_change_7d", "volume_pct_change_30d",
        "realized_daily_volatility",
        "price_deviation_5d", "price_deviation_30d",
        "downward_price_deviation_5d", "downward_price_deviation_30d",
    ]
    spillover = [
        f"{asset}_{col}"
        for asset in SPILLOVER_ASSETS
        for col in [
            "pct_change_1d", "pct_change_7d", "pct_change_30d",
            "volume_pct_change_1d", "volume_pct_change_7d", "volume_pct_change_30d",
            "realized_daily_volatility",
        ]
    ]
    return own + spillover


FEATURE_COLS = feature_columns()  # 11 own + 7 BTC + 7 ETH = 25 features


_SPILLOVER_CACHE: dict[str, pd.DataFrame] = {}


def _spillover_features(asset: str) -> pd.DataFrame:
    """BTC/ETH features are identical across every coin's dataset, so
    compute each spillover asset's feature table once and reuse it,
    instead of recomputing BTC/ETH rolling windows 7 times (once per coin).
    """
    if asset not in _SPILLOVER_CACHE:
        df = load_ohlcv(asset)
        _SPILLOVER_CACHE[asset] = add_rate_of_change_features(df, prefix=f"{asset}_")
    return _SPILLOVER_CACHE[asset]


def build_dataset(coin: str) -> pd.DataFrame:
    """Assemble one coin's modeling table: load -> label -> own features ->
    join spillover features -> restrict to Lee et al.'s sample window -> lag.

    REPLICATES: Lee et al. Section 4, "we structure data processing
    methodology to accurately predict the depegging of stablecoins for the
    subsequent day ... adjusting the dependent variable for a one-day
    forward prediction." i.e. day t's features predict day t+1's label --
    we implement that here by shifting `depeg` back one row into `target`,
    so each row already lines up (today's X, tomorrow's Y) by construction.
    """
    df = load_ohlcv(coin)
    df = add_label(df)

    merged = df[["date", "depeg"]].copy()
    merged = merged.merge(add_rate_of_change_features(df), on="date")
    merged = merged.merge(add_peg_deviation_features(df), on="date")
    for asset in SPILLOVER_ASSETS:
        merged = merged.merge(_spillover_features(asset), on="date")

    merged["target"] = merged["depeg"].shift(-1)

    # Drop the lookback margin now that every rolling window has had a
    # chance to fill in -- keep only Lee et al.'s actual 2022-2023 sample
    # period. (wluna's Yahoo history ends 2022-10-09 -- its "sample period"
    # is whatever overlap survives this filter, not the full 2 years.)
    merged = merged[(merged["date"] >= ANALYSIS_START) & (merged["date"] <= ANALYSIS_END)]

    # Drops: any remaining warm-up rows and the final row (no "tomorrow" to
    # predict yet).
    merged = merged.dropna(subset=FEATURE_COLS + ["target"]).reset_index(drop=True)
    merged["target"] = merged["target"].astype(int)
    return merged


def evaluate(coin: str, df: pd.DataFrame) -> list[dict]:
    """Fit LR + RF on one coin and score them.

    REPLICATES: Lee et al. Section 3.2.1 (models) + 3.2.2 (metrics) +
    Table 5 (the output format).

      - Models: Logistic Regression, Random Forest (their XGBoost is
        dropped -- out of scope per the workstream brief).
      - Split: stratified random train/test split on the label -- copied
        *deliberately*, not fixed, from their Section 4: "the division of
        the train dataset and test dataset does not employ a time-based
        split but instead utilizes stratified random sampling based on
        label Y." This is the exact design choice our own project's
        methodology (chronological, purged, no leakage across episodes)
        is a reaction against -- reproducing it as-is here is what lets a
        follow-up script show, on the same data, how different the
        chronological-split numbers turn out to be.
      - Metrics: accuracy/precision/recall/F1/specificity match their
        Table 5 exactly; ROC-AUC is *not* in their paper -- we add it
        because our workstream brief asks for it.
      - What we deliberately skip: SMOTE. Lee et al. apply it only to
        their low-depeg-rate coins (USDC, DAI) with a tuned 0.6 ratio; the
        brief asks for "a simple Logistic Regression and Random Forest",
        so we leave the class imbalance unfixed on purpose. Watching LR/RF
        fail on our own low-depeg-rate coins below is itself evidence for
        the shortcomings slide, not a bug to patch.
    """
    X = df[FEATURE_COLS]
    y = df["target"]

    if y.nunique() < 2:
        # e.g. a coin whose window never un-depegs (or never depegs) at all
        # -- there is no second class left to classify.
        return [{
            "coin": coin, "model": "-", "n_obs": len(df), "depeg_rate": y.mean(),
            "note": "single-class target, no model fit (skipped)",
        }]

    # stratify=y keeps each split's depeg rate close to the full dataset's,
    # matching their "structure and label proportions ... consistent with
    # those in the full dataset" -- but falls back to a plain random split
    # if a class is too rare to stratify on at all (e.g. a single depeg day
    # can't be split across train/test and still stratify).
    stratify = y if y.value_counts().min() >= 2 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RNG_SEED, stratify=stratify,
    )

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight=None),
        "Random Forest": RandomForestClassifier(n_estimators=300, random_state=RNG_SEED),
    }

    rows = []
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else y_pred

        # Confusion matrix (Lee et al. Table 2) -> the 5 metrics in their Table 5.
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else np.nan
        try:
            auc = roc_auc_score(y_test, y_proba)
        except ValueError:
            auc = np.nan  # y_test happened to land all-one-class after the split

        rows.append({
            "coin": coin,
            "model": name,
            "n_obs": len(df),
            "depeg_rate": round(y.mean(), 4),
            "accuracy": round(accuracy_score(y_test, y_pred), 3),
            "precision": round(precision_score(y_test, y_pred, zero_division=0), 3),
            "recall": round(recall_score(y_test, y_pred, zero_division=0), 3),
            "f1": round(f1_score(y_test, y_pred, zero_division=0), 3),
            "specificity": round(specificity, 3) if not np.isnan(specificity) else np.nan,
            "roc_auc": round(auc, 3) if not np.isnan(auc) else np.nan,
        })
    return rows


def main():
    all_rows = []
    episode_summary = []
    for coin in COINS:
        df = build_dataset(coin)
        n_depeg = int(df["target"].sum())
        episode_summary.append({
            "coin": coin,
            "n_obs": len(df),
            "date_start": df["date"].min().date(),
            "date_end": df["date"].max().date(),
            "n_depeg_days": n_depeg,
            "depeg_rate": round(df["target"].mean(), 4),
        })
        all_rows.extend(evaluate(coin, df))

    episode_df = pd.DataFrame(episode_summary)
    results_df = pd.DataFrame(all_rows)

    print("=" * 90)
    print("Depeg label summary (dynamic threshold, per-coin)")
    print("=" * 90)
    print(episode_df.to_string(index=False))
    # Sanity-check worth reading, not just running: usdt/usdc/dai/pax/busd
    # should land in a plausible <30% depeg-rate range, same shape as Lee
    # et al.'s Table 4 (now over the *same* ~2-year window as their paper).
    # ustc/wluna instead show very high depeg rates -- NOT because these
    # coins generated hundreds of independent crisis days, but because most
    # of their remaining history sits after the May 9, 2022 UST/LUNA
    # collapse, so nearly every day from then on is trivially still
    # "depegged". That is exactly the pseudo-replication trap the professor
    # flagged: one continuous episode being counted as if it were hundreds
    # of independent observations. Lee et al.'s stratified-random/daily-
    # binary design has no way to tell the difference -- which is the point
    # of this whole exercise, not a flaw in this script.
    print()
    print("=" * 90)
    print("Model performance (stratified random split, faithful to Lee et al. 2025 Table 5)")
    print("=" * 90)
    print(results_df.to_string(index=False))

    out_dir = os.path.join(REPO_ROOT, "models")
    episode_df.to_csv(os.path.join(out_dir, "lee_replication_label_summary.csv"), index=False)
    results_df.to_csv(os.path.join(out_dir, "lee_replication_results.csv"), index=False)


if __name__ == "__main__":
    main()
