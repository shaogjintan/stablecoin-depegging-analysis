"""Dump fitted LogReg coefficients (all saved folds) to a CSV.
Reads saved models only; no retraining. Records tag/tune/C/penalty per fold
so tuned (lr01) and fixed-config runs are never confused."""
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))  # lets joblib find the `modelling` package
MODELS = REPO / "results" / "models"
OUT = REPO / "reports" / "logreg_coefficients.csv"

rows = []
for path in sorted(MODELS.glob("logreg_t*_h*_*.joblib")):
    meta_path = path.with_suffix(".json")
    if not meta_path.exists():
        continue
    meta = json.loads(meta_path.read_text())
    model = joblib.load(path)
    names = list(model.cols_)
    coefs = np.asarray(model.clf_.coef_).ravel()
    if len(names) != len(coefs):
        print(f"skip {path.name}: length mismatch")
        continue
    mp = meta.get("model_params") or {}
    for name, coef in zip(names, coefs):
        rows.append({
            "file": path.name,
            "tier": meta.get("tier"),
            "horizon": meta.get("horizon"),
            "year": meta.get("year"),
            "tag": meta.get("tag"),
            "tune": meta.get("tune"),
            "C": mp.get("C"),
            "penalty": mp.get("penalty"),
            "feature": name,
            "is_onchain": "logdiff" in name,
            "coef": coef,
            "abs_coef": abs(coef),
        })

df = pd.DataFrame(rows)
OUT.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(OUT, index=False)
print(f"Saved {OUT} ({len(df)} rows)")
t2 = df[df.tier == 2]
for (h, y, tag), g in t2.groupby(["horizon", "year", "tag"]):
    oc = g[g.is_onchain]
    mk = g[~g.is_onchain]
    print(f"h{h} {y} {tag}: on-chain nonzero {int((oc.coef != 0).sum())}/{len(oc)}, "
          f"max on-chain |coef| {oc.abs_coef.max():.3f}, "
          f"max market |coef| {mk.abs_coef.max():.3f}")
