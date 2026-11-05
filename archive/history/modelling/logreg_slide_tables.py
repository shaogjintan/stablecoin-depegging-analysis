"""Build the two LogReg slide tables (tuned lr01) from committed files.

1. Tuning: chosen C/penalty per fold  -> reports/logreg_tuning_summary.csv
2. Coefficients: h=24 Tier 2 per test year: top-3 market coefficients,
   on-chain nonzero count, max on-chain coefficient
   -> reports/logreg_coef_slide_h24_t2.csv
No retraining; reads reports/logreg_tuning/*.json and reports/logreg_coefficients.csv.
"""
import json
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
TUNE_DIR = REPO / "reports" / "logreg_tuning"
COEF_CSV = REPO / "reports" / "logreg_coefficients.csv"

# ---- 1. tuning summary ----
rows = []
for p in sorted(TUNE_DIR.glob("logreg_t*_h*_*.json")):
    meta = json.loads(p.read_text())
    mp = meta.get("model_params")
    if not mp:
        print(f"{p.name}: no model_params; keys = {list(meta)[:15]}")
        continue
    stem = p.stem.split("_")  # logreg, t1, h24, 2021
    rows.append({
        "tier": int(stem[1][1:]), "horizon": int(stem[2][1:]), "year": int(stem[3]),
        "tag": meta.get("tag"), "tune": meta.get("tune"),
        "C": mp.get("C"), "penalty": mp.get("penalty"),
    })
tune = pd.DataFrame(rows)
tune.to_csv(REPO / "reports" / "logreg_tuning_summary.csv", index=False)
print(f"Tuning folds read: {len(tune)}")
print(tune.groupby(["C", "penalty"]).size().rename("n_folds").to_string())
print("\nBy slice:")
for (t, h), g in tune.groupby(["tier", "horizon"]):
    print(f"T{t} h{h}: " + "; ".join(
        f"{r.year}: C={r.C}, {r.penalty}" for r in g.itertuples()))

# ---- 2. coefficient slide (h=24, Tier 2) ----
df = pd.read_csv(COEF_CSV)
d = df[(df.tier == 2) & (df.horizon == 24)]
out = []
for year, g in d.groupby("year"):
    mk = g[~g.is_onchain].nlargest(3, "abs_coef")
    oc = g[g.is_onchain]
    top_oc = oc.loc[oc.abs_coef.idxmax()]
    out.append({
        "year": year, "tag": g.tag.iloc[0], "C": g.C.iloc[0],
        "penalty": g.penalty.iloc[0],
        "top3_market": "; ".join(f"{r.feature} ({r.coef:.2f})" for r in mk.itertuples()),
        "onchain_nonzero": f"{int((oc.coef != 0).sum())}/{len(oc)}",
        "max_onchain": f"{top_oc.feature} ({top_oc.coef:.2f})",
    })
slide = pd.DataFrame(out)
slide.to_csv(REPO / "reports" / "logreg_coef_slide_h24_t2.csv", index=False)
print("\nCoefficient slide (tuned lr01, h=24, Tier 2):")
print(slide.to_string(index=False))
