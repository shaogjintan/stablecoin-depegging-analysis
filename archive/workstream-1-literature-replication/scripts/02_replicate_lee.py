#!/usr/bin/env python3
"""
TASK 2 v4: Replicate Lee, Chiu & Hsieh (2025) on the paper's full window
(2022-01-01 .. 2023-12-31) plus the May-2022 sub-window from the project spec.

Depeg definitions (paper PDF p.5):
  dynamic: Thresh_D = 1 - 10/V^alpha, Thresh_U = 1 + 10/V^alpha,
           V = rolling 30-day sum of trading volume (USD), alpha = 1/3.
  fixed:   simplified +/-1% band (low < 0.99 or high > 1.01).
Label: Y_t = 1 if the coin depegs on the NEXT day (1-day forward).

Models: Logistic Regression, Random Forest, XGBoost (paper's spec).
SMOTE (0.6) on the training set only when depeg ratio < 10% (paper's rule).
Splits: random stratified 70/30 (their method) | chronological
(2022 train -> 2023 test) | may-window (Apr1-May7 -> May8-Jun30 2022).
Outputs: outputs/replication_results.csv, outputs/feature_importance.csv
"""
import warnings
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, average_precision_score)
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

warnings.filterwarnings("ignore")

STABLES = ["USDT", "USDC", "BUSD", "DAI"]
CROSS = ["BTC", "ETH"]
SEED = 42
ALPHA = 1 / 3

PUBLISHED = [
    ("Logistic Regression", "BUSD", 0.869, 0.783, 0.735, 0.758),
    ("Random Forest",       "BUSD", 0.891, 0.788, 0.837, 0.812),
    ("XGBoost",             "BUSD", 0.880, 0.792, 0.776, 0.784),
    ("Logistic Regression", "USDT", 0.829, 0.773, 0.405, 0.531),
    ("Random Forest",       "USDT", 0.886, 0.806, 0.690, 0.744),
    ("XGBoost",             "USDT", 0.897, 0.800, 0.762, 0.780),
    ("Logistic Regression", "USDC", 0.928, 0.167, 0.500, 0.250),
    ("Random Forest",       "USDC", 0.994, 1.000, 0.750, 0.857),
    ("XGBoost",             "USDC", 0.982, 0.600, 0.750, 0.667),
    ("Logistic Regression", "DAI",  0.929, 0.214, 0.750, 0.333),
    ("Random Forest",       "DAI",  0.982, 0.667, 0.500, 0.571),
    ("XGBoost",             "DAI",  0.982, 0.667, 0.500, 0.571),
]


def load_hourly(path):
    df = pd.read_csv(path, parse_dates=["timestamp"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df


def daily_ohlcv(hourly):
    out = []
    for coin, sub in hourly.groupby("coin"):
        sub = sub.set_index("timestamp").sort_index()
        out.append(pd.DataFrame({
            "coin": coin,
            "open":  sub["price"].resample("D").first(),
            "high":  sub["price"].resample("D").max(),
            "low":   sub["price"].resample("D").min(),
            "close": sub["price"].resample("D").last(),
            "volume": sub["volume"].resample("D").sum(),
        }))
    return pd.concat(out).reset_index()


def grobys_vol(o, h, l, c):
    x = np.abs(np.log(h / c) * np.log(h / o) + np.log(l / c) * np.log(l / o))
    return np.sqrt(x)


def add_labels(daily):
    rows = []
    for coin in STABLES:
        d = daily[daily["coin"] == coin].set_index("timestamp").sort_index().copy()
        V = d["volume"].rolling(30, min_periods=30).sum()
        thresh = 10.0 / np.power(V, ALPHA)
        d["thresh_D"], d["thresh_U"] = 1.0 - thresh, 1.0 + thresh
        d["depeg_dynamic"] = ((d["low"] <= d["thresh_D"]) | (d["high"] >= d["thresh_U"])).astype(int)
        d["depeg_fixed"] = ((d["low"] < 0.99) | (d["high"] > 1.01)).astype(int)
        d["coin"] = coin
        rows.append(d)
    out = pd.concat(rows)
    out["Y_dynamic"] = out.groupby("coin")["depeg_dynamic"].shift(-1)
    out["Y_fixed"] = out.groupby("coin")["depeg_fixed"].shift(-1)
    return out.reset_index()


FEATURES = ["pc_24h", "pc_7d", "pc_30d", "vol_pc_24h", "vol_pc_7d", "vol_pc_30d",
            "real_vol", "dev_5d", "dev_30d", "BTC_pc_24h", "BTC_pc_7d", "BTC_real_vol",
            "ETH_pc_24h"]


def build_features(daily):
    feats = {}
    for coin in STABLES + CROSS:
        sub = daily[daily["coin"] == coin].set_index("timestamp").sort_index()
        f = pd.DataFrame(index=sub.index)
        f["pc_24h"] = sub["close"].pct_change(1)
        f["pc_7d"] = sub["close"].pct_change(7)
        f["pc_30d"] = sub["close"].pct_change(30)
        f["vol_pc_24h"] = sub["volume"].pct_change(1)
        f["vol_pc_7d"] = sub["volume"].pct_change(7)
        f["vol_pc_30d"] = sub["volume"].pct_change(30)
        f["real_vol"] = grobys_vol(sub["open"], sub["high"], sub["low"], sub["close"])
        f["dev_5d"] = (sub["close"] - 1.0).pow(2).rolling(5).std()
        f["dev_30d"] = (sub["close"] - 1.0).pow(2).rolling(30).std()
        feats[coin] = f
    for coin in STABLES:
        own = feats[coin].copy()
        for suffix in ["pc_24h", "pc_7d", "real_vol"]:
            own[f"BTC_{suffix}"] = feats["BTC"][suffix]
        own[f"ETH_pc_24h"] = feats["ETH"]["pc_24h"]
        own["coin"] = coin
        feats[coin] = own
    return feats


def make_dataset(daily, label_col, crop_start=None, crop_end=None):
    feats = build_features(daily)
    labeled = add_labels(daily)
    frames = []
    for coin in STABLES:
        f = feats[coin].copy()
        lab = labeled[labeled["coin"] == coin].set_index("timestamp")
        f["Y"] = lab[label_col]
        f["coin"] = coin
        frames.append(f)
    df = pd.concat(frames).reset_index().rename(columns={"timestamp": "date"})
    if crop_start:
        df = df[df["date"] >= pd.Timestamp(crop_start, tz="UTC")]
    if crop_end:
        df = df[df["date"] < pd.Timestamp(crop_end, tz="UTC")]
    df = df.replace([np.inf, -np.inf], np.nan)
    return df.dropna(subset=FEATURES + ["Y"])


def make_models(pos_scale=1.0):
    return {
        "Logistic Regression": LogisticRegression(
            C=1.0, penalty="l2", class_weight="balanced", max_iter=2000, random_state=SEED),
        "Random Forest": RandomForestClassifier(
            n_estimators=100, max_depth=5, class_weight="balanced", random_state=SEED, n_jobs=-1),
        "XGBoost": XGBClassifier(
            n_estimators=100, max_depth=4, scale_pos_weight=pos_scale,
            random_state=SEED, eval_metric="logloss"),
    }


def evaluate(y_true, y_pred, y_prob):
    out = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }
    if len(set(y_true)) > 1:
        out["roc_auc"] = roc_auc_score(y_true, y_prob)
        out["pr_auc"] = average_precision_score(y_true, y_prob)
    else:
        out["roc_auc"] = out["pr_auc"] = np.nan
    return out


def constant_zero_baseline(yte):
    m = evaluate(yte, np.zeros(len(yte), dtype=int), np.zeros(len(yte)))
    m["roc_auc"] = 0.5 if len(set(yte)) > 1 else np.nan
    m["pr_auc"] = yte.mean() if len(set(yte)) > 1 else np.nan
    return m


def smote_resample(Xtr, ytr):
    pos_ratio = ytr.mean()
    if pos_ratio < 0.10 and int(ytr.sum()) >= 2:
        k = max(1, min(5, int(ytr.sum()) - 1))
        try:
            Xtr, ytr = SMOTE(sampling_strategy=0.6, k_neighbors=k,
                             random_state=SEED).fit_resample(Xtr, ytr)
        except Exception:
            pass
    return Xtr, ytr


def run_model(model, Xtr, ytr, Xte, yte):
    counts = {"n_train": len(ytr), "n_train_pos": int(ytr.sum()),
              "n_test": len(yte), "n_test_pos": int(yte.sum())}
    if int(ytr.sum()) == 0:
        m = constant_zero_baseline(yte)
        m["note"] = "no positives in train"
    elif len(set(ytr)) < 2:
        m = {k: np.nan for k in ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]}
        m["note"] = "degenerate"
    else:
        Xtr, ytr = smote_resample(Xtr, ytr)
        model.fit(Xtr, ytr)
        pred, prob = model.predict(Xte), model.predict_proba(Xte)[:, 1]
        m = evaluate(yte, pred, prob)
        m["note"] = "fitted"
    m.update(counts)
    return m


def split_random(sub):
    X, y = sub[FEATURES], sub["Y"]
    try:
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, stratify=y, random_state=SEED)
    except ValueError:
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=SEED)
    return Xtr, ytr, Xte, yte


def main():
    hourly = load_hourly("data/hourly_prices_full_2022_2023.csv")
    daily = daily_ohlcv(hourly)
    rows = []

    for variant in ["dynamic", "fixed"]:
        data = make_dataset(daily, f"Y_{variant}")  # full window (from Feb 2022, 30d warmup)
        may = make_dataset(daily, f"Y_{variant}", crop_start="2022-04-01",
                            crop_end="2022-07-01")
        print(f"\n########## THRESHOLD VARIANT: {variant} ##########")
        for name, ds in [("FULL 2022-2023", data), ("MAY 2022 window", may)]:
            print(f"--- {name}: {len(ds)} days, depeg-days={int(ds['Y'].sum())} "
                  f"({ds['Y'].mean():.2%})")
            per = ds.groupby("coin")["Y"].agg(["sum", "count"])
            per["ratio"] = (per["sum"] / per["count"]).round(3)
            print(per.to_string())

        for coin in STABLES:
            sub = data[data["coin"] == coin].reset_index(drop=True)
            pos_scale = max(1.0, (len(sub) - sub["Y"].sum()) / max(sub["Y"].sum(), 1))
            for split_name, parts in [
                    ("random", split_random(sub)),
                    ("chrono_2023", None),
            ]:
                if split_name == "chrono_2023":
                    mask = sub["date"] < pd.Timestamp("2023-01-01", tz="UTC")
                    Xtr, ytr = sub.loc[mask, FEATURES], sub.loc[mask, "Y"]
                    Xte, yte = sub.loc[~mask, FEATURES], sub.loc[~mask, "Y"]
                else:
                    Xtr, ytr, Xte, yte = parts
                for mname, model in make_models(pos_scale).items():
                    m = run_model(model, Xtr, ytr, Xte, yte)
                    rows.append({"Model": mname, "Coin": coin, "Split": split_name,
                                 "Threshold": variant, "Window": "full", **m})
            # may-window splits (project spec)
            sub_m = may[may["coin"] == coin].reset_index(drop=True)
            mask = sub_m["date"] <= pd.Timestamp("2022-05-07", tz="UTC")
            Xtr, ytr = sub_m.loc[mask, FEATURES], sub_m.loc[mask, "Y"]
            Xte, yte = sub_m.loc[~mask, FEATURES], sub_m.loc[~mask, "Y"]
            for mname, model in make_models(pos_scale).items():
                m = run_model(model, Xtr, ytr, Xte, yte)
                rows.append({"Model": mname, "Coin": coin, "Split": "chrono_may",
                             "Threshold": variant, "Window": "may", **m})

        # pooled, full window
        pos_scale = max(1.0, (len(data) - data["Y"].sum()) / max(data["Y"].sum(), 1))
        for split_name, parts in [("random", split_random(data)),
                                  ("chrono_2023", None)]:
            if split_name == "chrono_2023":
                mask = data["date"] < pd.Timestamp("2023-01-01", tz="UTC")
                Xtr, ytr = data.loc[mask, FEATURES], data.loc[mask, "Y"]
                Xte, yte = data.loc[~mask, FEATURES], data.loc[~mask, "Y"]
            else:
                Xtr, ytr, Xte, yte = parts
            for mname, model in make_models(pos_scale).items():
                m = run_model(model, Xtr, ytr, Xte, yte)
                rows.append({"Model": mname, "Coin": "POOLED", "Split": split_name,
                             "Threshold": variant, "Window": "full", **m})

        if variant == "dynamic":
            Xtr, ytr, Xte, yte = split_random(data)
            Xtr_s, ytr_s = smote_resample(Xtr, ytr)
            rf = RandomForestClassifier(n_estimators=100, max_depth=5,
                                        class_weight="balanced", random_state=SEED, n_jobs=-1)
            rf.fit(Xtr_s, ytr_s)
            imp = pd.Series(rf.feature_importances_, index=FEATURES).sort_values(ascending=False)
            imp.to_csv("outputs/feature_importance.csv", header=["importance"])
            print("\nTop 10 feature importances (pooled RF, random split, dynamic thr., full window):")
            for rank, (f, v) in enumerate(imp.head(10).items(), 1):
                print(f"  {rank:>2}. {f:<16} {v:.4f}")

    res = pd.DataFrame(rows)
    pub = pd.DataFrame(PUBLISHED, columns=["Model", "Coin", "pub_accuracy",
                                           "pub_precision", "pub_recall", "pub_f1"])
    out = res.merge(pub, on=["Model", "Coin"], how="left")
    out.to_csv("outputs/replication_results.csv", index=False)
    print(f"\nsaved outputs/replication_results.csv ({len(out)} rows)")

    print("\n===== PUBLISHED vs OURS — dynamic threshold, random split, full window =====")
    for model, coin, acc, prec, rec, f1 in PUBLISHED:
        ours = next((r for r in rows if r["Model"] == model and r["Coin"] == coin
                     and r["Split"] == "random" and r["Threshold"] == "dynamic"
                     and r["Window"] == "full"), None)
        if ours and ours["note"] == "fitted":
            print(f"  {model:<20} {coin:>4}:  pub Acc={acc:.3f} Prec={prec:.3f} Rec={rec:.3f} "
                  f"F1={f1:.3f}  |  ours Acc={ours['accuracy']:.3f} Prec={ours['precision']:.3f} "
                  f"Rec={ours['recall']:.3f} F1={ours['f1']:.3f}")
        else:
            print(f"  {model:<20} {coin:>4}:  pub Acc={acc:.3f} Prec={prec:.3f} Rec={rec:.3f} "
                  f"F1={f1:.3f}  |  ours: {ours['note'] if ours else 'n/a'}")

    print("\n===== RANDOM vs CHRONOLOGICAL (pooled, dynamic threshold, full window) =====")
    pool = out[(out["Coin"] == "POOLED") & (out["Threshold"] == "dynamic") & (out["Window"] == "full")]
    print(pool[["Model", "Split", "accuracy", "precision", "recall", "f1", "roc_auc",
                "pr_auc", "n_train_pos", "note"]].to_string(
        index=False, float_format=lambda v: f"{v:.3f}"))

    print("\n===== PER-COIN (dynamic threshold, full window) =====")
    per = out[(out["Coin"] != "POOLED") & (out["Threshold"] == "dynamic") & (out["Window"] == "full")]
    print(per[["Model", "Coin", "Split", "accuracy", "precision", "recall", "f1",
               "roc_auc", "n_train_pos", "note"]].to_string(
        index=False, float_format=lambda v: f"{v:.3f}"))


if __name__ == "__main__":
    main()
