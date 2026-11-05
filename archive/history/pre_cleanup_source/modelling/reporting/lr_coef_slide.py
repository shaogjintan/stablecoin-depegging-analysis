"""
LR "Adding on-chain" slide table from a verified corrected LR run.

Same layout as modelling/logreg_slide_tables.py (which reads the legacy lr01
files), but sourced from a corrected_lr_rf run directory: per test year, the
C chosen by calendar-year CV, the top-3 market coefficients, how many of the
12 on-chain coefficients are non-zero, and the largest on-chain coefficient.
Coefficients are standardised L1 coefficients of the scoring fit (inner_train).

The run must have passed `python -m modelling.verify_lr_rf_run` first.

INPUT (<run-dir>, e.g. runs/LR-NEW)
  run.json, verification.json, lr_selected_c.csv, importance_per_fit.csv

OUTPUT (reports/lr_corrected/)
  coef_slide_<config>_h<h>.csv

Usage (from the repo root):
    python modelling/reporting/lr_coef_slide.py --run-dir runs/LR-NEW
    python modelling/reporting/lr_coef_slide.py --run-dir runs/LR-NEW --config market_onchain --horizon 24
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]


def slide_table(run_dir: Path, config: str, horizon: int) -> pd.DataFrame:
    info = json.loads((run_dir / "run.json").read_text())
    verified = json.loads((run_dir / "verification.json").read_text())
    if verified.get("status") != "passed":
        raise ValueError(f"{run_dir} has not passed verify_lr_rf_run")

    coefs = pd.read_csv(run_dir / "importance_per_fit.csv")
    coefs = coefs[(coefs["model"] == "logreg") & (coefs["config"] == config)
                  & (coefs["horizon_hours"] == horizon)]
    chosen = pd.read_csv(run_dir / "lr_selected_c.csv")
    chosen = chosen[(chosen["config"] == config) & (chosen["horizon_hours"] == horizon)]
    if coefs.empty or chosen.empty:
        raise ValueError(f"No logreg fits for {config} h{horizon} in {run_dir}")
    c_by_year = chosen.set_index("test_year")["selected_C"]

    rows = []
    for year, g in coefs.groupby("test_year"):
        g = g.assign(abs_value=g["value"].abs())
        market = g[~g["is_onchain"]].nlargest(3, "abs_value")
        onchain = g[g["is_onchain"]]
        top_onchain = onchain.loc[onchain["abs_value"].idxmax()]
        rows.append({
            "test_year": year,
            "C": c_by_year[year],
            "top3_market": "; ".join(f"{r.feature} ({r.value:.2f})"
                                     for r in market.itertuples()),
            "onchain_nonzero": f"{int((onchain['value'] != 0).sum())}/{len(onchain)}",
            "max_onchain": f"{top_onchain.feature} ({top_onchain.value:.2f})",
            "top_market_abs": round(market["abs_value"].iloc[0], 4),
            "max_onchain_abs": round(top_onchain.abs_value, 4),
            "run_id": info["run_id"],
        })
    return pd.DataFrame(rows)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--config", default="market_onchain")
    parser.add_argument("--horizon", type=int, default=24)
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "reports/lr_corrected")
    args = parser.parse_args(argv)

    table = slide_table(args.run_dir.resolve(), args.config, args.horizon)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    path = args.out_dir / f"coef_slide_{args.config}_h{args.horizon}.csv"
    table.to_csv(path, index=False)
    print(table.drop(columns="run_id").to_string(index=False))
    print(f"saved {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
