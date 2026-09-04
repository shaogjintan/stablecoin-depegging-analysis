"""
Replication of the core result in:
    Lee, Y.-H., Chiu, Y.-F., & Hsieh, M.-H. (2025). Stablecoin depegging risk
    prediction. Pacific-Basin Finance Journal, 90, 102640.

WHAT WE ARE REPLICATING, EXACTLY
---------------------------------
Lee et al. is a full pipeline: 4 coins (USDT/USDC/BUSD/DAI) x 66 features
(price/volume, market-cap/supply, sentiment, volatility) x 3 models
(LR/RF/XGBoost) x SMOTE-tuned stratified split -> Table 5 (F1-led comparison).

Our workstream brief scopes this down to: "Run a simple Logistic Regression
and Random Forest using daily price/volatility indicators. Produce a table
showing their baseline accuracy/ROC-AUC." So this script replicates three
specific, load-bearing pieces of their paper, and nothing else:

  (a) THE LABEL (Section 3.1, the Y equation).
      Their single biggest methodological contribution isn't a model, it's
      how they define "depegged" at all: a threshold that tightens for
      high-volume coins and loosens for low-volume ones, rather than one
      fixed band for every coin. We reproduce this formula exactly.

  (b) A REDUCED SLICE OF THEIR FEATURES (Table 1, blocks 1 and 4 only).
      Their Table 1 has 4 feature blocks: (1) price/volume change rates,
      (2) market-cap/circulating-supply/total-supply change rates,
      (3) sentiment indices, (4) volatility indicators. We only have
      raw OHLC + volume in our data (no market-cap/supply series, no
      sentiment feeds), so we replicate blocks (1) and (4) only -- which is
      exactly what "daily price/volatility indicators" in the brief means.
      We also don't have BTC/ETH cross-asset versions of these features
      (their model triples the feature count by repeating each block for
      BTC and ETH); we skip that entirely.

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
  - BTC/ETH spillover features, market-cap/supply features, sentiment
    indices (Fear & Greed, Sentiment, Awareness) -- data we don't have,
    and sentiment turned out not to matter in their own results anyway
    (see their Table 6 -- no sentiment variable makes the top 15).
  - XGBoost and their SMOTE hyperparameter search -- out of scope per brief.
  - Their exact coin roster. They use USDT/USDC/BUSD/DAI; we don't have
    BUSD in our raw dataset (ERC20-stablecoins/), so we run their exact
    method over our project's actual coin universe instead: usdt, usdc,
    dai, pax (their DAI/USDC/USDT analogues), plus ustc and wluna as a
    bonus -- the two algorithmic coins Lee et al. excluded, which happen
    to be the ones that actually failed. Seeing their method choke on
    those two is itself a useful result (see the module docstring at the
    bottom of main()).

DATA SOURCES
------------
  - Price (OHLC, daily): ERC20-stablecoins/price_data/ -- the project's
    existing raw dataset, already profiled in models/eda.ipynb.
  - Volume: data/volume/*_volume_data.csv, which we pulled from Yahoo
    Finance ourselves. CoinGecko's free API refuses anything older than
    365 days now, and Lee et al.'s threshold formula needs trailing
    30-day *volume*, which the local OHLC files don't contain at all --
    so this was a genuine gap in ERC20-stablecoins/, not a design choice.
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
# repo root's parent -> DSE4101/, sibling of ERC20-stablecoins/
PROJECT_ROOT = os.path.dirname(REPO_ROOT)
PRICE_DIR = os.path.join(PROJECT_ROOT, "ERC20-stablecoins", "price_data", "price_data")
VOLUME_DIR = os.path.join(REPO_ROOT, "data", "volume")

# Our project's coin universe (matches ERC20-stablecoins/price_data), not
# Lee et al.'s USDT/USDC/BUSD/DAI -- see docstring above for why.
COINS = ["usdt", "usdc", "dai", "pax", "ustc", "wluna"]

# --- Carey (2023) / Kaiko dynamic-threshold constants, as used by Lee et al. ---
# ThreshD/U = 1 -/+ K / V_monthly^ALPHA. Lee et al. cite ALPHA = 1/3 as Carey's
# "optimal setting" and K = 10; we take both as given rather than re-deriving them.
ALPHA = 1 / 3
K = 10

RNG_SEED = 42     # fixed seed so the "faithful" stratified split is reproducible
TEST_SIZE = 0.3   # Lee et al. don't state their exact split ratio; 70/30 is standard


def load_coin_frame(coin: str) -> pd.DataFrame:
    """Merge daily OHLC (ours) with daily volume (Yahoo) for one coin.

    Lee et al.'s underlying data is genuinely one merged table per coin --
    CoinMarketCap already gives them price *and* volume together. We have
    to build that merge ourselves because our price and volume come from
    two different sources with two different date formats (unix seconds
    vs. an ISO date column).
    """
    price = pd.read_csv(os.path.join(PRICE_DIR, f"{coin}_price_data.csv"))
    price["date"] = pd.to_datetime(price["timestamp"], unit="s", utc=True).dt.tz_localize(None)

    volume = pd.read_csv(os.path.join(VOLUME_DIR, f"{coin}_volume_data.csv"))
    volume["date"] = pd.to_datetime(volume["date"])
    volume = volume[["date", "volume"]]

    df = price.merge(volume, on="date", how="left").sort_values("date").reset_index(drop=True)
    # A few coins (e.g. wluna) have gaps in Yahoo's volume history where the
    # pair was thin/delisted; forward-fill rather than drop those days, since
    # dropping would silently shrink our already-small sample.
    df["volume"] = df["volume"].ffill()
    return df


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
    to estimate that day's realized volatility -- no rolling window needed,
    which is why (unlike the *_deviation features below) it doesn't cost us
    any extra warm-up days.
    """
    po, ph, pl, pc = row["open"], row["high"], row["low"], row["close"]
    if min(po, ph, pl, pc) <= 0:
        return np.nan
    val = np.log(ph / pc) * np.log(ph / po) + np.log(pl / pc) * np.log(pl / po)
    # Floating point can push a near-zero value fractionally negative; clip
    # before sqrt rather than let it silently become NaN.
    return np.sqrt(max(val, 0.0))


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """REPLICATES: Lee et al. Table 1, blocks (1) "trading price/volume
    change rate" and (4) "volatility indicators" -- the two blocks we can
    build from OHLC + volume alone. Blocks (2) market-cap/supply and (3)
    sentiment are skipped; see module docstring.

    All of these are computed *contemporaneously* (using data through day
    t) -- the "lag by one period" step happens later in build_dataset(),
    by shifting the *target* forward instead of shifting these features
    backward. Same effect, simpler to get right.
    """
    df = df.copy()
    close = df["close"]
    vol = df["volume"]

    # --- Block 1a: trading price change rate ---
    # Lee et al. use 1h/24h/7d/30d price % change; we only have daily data,
    # so 24h collapses to a plain 1-day change and 1h has no daily analogue.
    df["pct_change_1d"] = close.pct_change(1)
    df["pct_change_7d"] = close.pct_change(7)
    df["pct_change_30d"] = close.pct_change(30)

    # --- Block 1b: trading volume change rate ---
    # Same idea, applied to volume instead of price (their 24h/7d/30d volume
    # % change).
    df["volume_pct_change_1d"] = vol.pct_change(1)
    df["volume_pct_change_7d"] = vol.pct_change(7)
    df["volume_pct_change_30d"] = vol.pct_change(30)

    # --- Block 4a: Realized Daily Volatility (Rogers-Satchell) ---
    df["realized_daily_volatility"] = df.apply(rogers_satchell, axis=1)

    # --- Block 4b/c: Price Deviation and Downward Price Deviation ---
    # Kwon et al. (2023)'s definitions, as cited by Lee et al.:
    #   Price_Deviation_T   = sqrt( mean_{t-T..t}[ (P_close - 1)^2 ] )
    #   Downward_Deviation_T = sqrt( mean_{t-T..t}[ min(P_close - 1, 0)^2 ] )
    # "1" here is the $1 peg -- these measure RMS distance from the peg
    # (Price Deviation) vs. RMS distance *only counting downside misses*
    # (Downward Deviation), each over a 5-day and a 30-day rolling window,
    # exactly as Lee et al. set T.
    dev = (close - 1) ** 2
    df["price_deviation_5d"] = np.sqrt(dev.rolling(5).mean())
    df["price_deviation_30d"] = np.sqrt(dev.rolling(30).mean())

    down_dev = close.apply(lambda p: min(p - 1, 0) ** 2)
    df["downward_price_deviation_5d"] = np.sqrt(down_dev.rolling(5).mean())
    df["downward_price_deviation_30d"] = np.sqrt(down_dev.rolling(30).mean())

    return df


# The 11 features actually fed to the models -- our stand-in for Lee et al.'s
# 66-variable Table 1 (reduced to what raw OHLC + volume alone can produce;
# see module docstring for exactly what's dropped and why).
FEATURE_COLS = [
    "pct_change_1d", "pct_change_7d", "pct_change_30d",
    "volume_pct_change_1d", "volume_pct_change_7d", "volume_pct_change_30d",
    "realized_daily_volatility",
    "price_deviation_5d", "price_deviation_30d",
    "downward_price_deviation_5d", "downward_price_deviation_30d",
]


def build_dataset(coin: str) -> pd.DataFrame:
    """Assemble one coin's modeling table: load -> label -> features -> lag.

    REPLICATES: Lee et al. Section 4, "we structure data processing
    methodology to accurately predict the depegging of stablecoins for the
    subsequent day ... adjusting the dependent variable for a one-day
    forward prediction." i.e. day t's features predict day t+1's label --
    we implement that here by shifting `depeg` back one row into `target`,
    so each row already lines up (today's X, tomorrow's Y) by construction.
    """
    df = load_coin_frame(coin)
    df = add_label(df)
    df = add_features(df)
    df["target"] = df["depeg"].shift(-1)
    # Drops: the 29-day label warm-up, the 30-day feature warm-up, and the
    # final row (no "tomorrow" to predict yet).
    df = df.dropna(subset=FEATURE_COLS + ["target"]).reset_index(drop=True)
    df["target"] = df["target"].astype(int)
    return df


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
        fail on our own low-depeg-rate coins (usdc/dai/pax below) is
        itself evidence for the shortcomings slide, not a bug to patch.
    """
    X = df[FEATURE_COLS]
    y = df["target"]

    if y.nunique() < 2:
        # e.g. wluna: depeg rate is 100% in our window (see main()'s note on
        # why) -- there is no "undepegged" class left to classify at all.
        return [{
            "coin": coin, "model": "-", "n_obs": len(df), "depeg_rate": y.mean(),
            "note": "single-class target, no model fit (skipped)",
        }]

    # stratify=y keeps each split's depeg rate close to the full dataset's,
    # matching their "structure and label proportions ... consistent with
    # those in the full dataset" -- but falls back to a plain random split
    # if a class is too rare to stratify on at all (e.g. usdc's single
    # depeg day can't be split across train/test and still stratify).
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
    # Sanity-check worth reading, not just running: usdt/usdc/dai/pax land in
    # a plausible <30% depeg-rate range, same shape as Lee et al.'s Table 4.
    # ustc/wluna instead show ~100% depeg rates -- NOT because these coins
    # generated ~180 independent crisis days, but because our fixed window
    # (2022-05-02 to 2022-11-01) sits almost entirely *after* the May 9 UST
    # collapse, so nearly every remaining day is trivially still "depegged".
    # That is exactly the pseudo-replication trap the professor flagged: one
    # continuous episode being counted as if it were hundreds of independent
    # observations. Lee et al.'s stratified-random/daily-binary design has
    # no way to tell the difference -- which is the point of this whole
    # exercise, not a flaw in this script.
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
