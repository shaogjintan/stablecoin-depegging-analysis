"""Export calibration evidence directly from saved final fits; no training."""
import argparse
import json
from pathlib import Path
import pandas as pd
from modelling.utils.artifacts import evidence_dir
from modelling.reporting.model_results import RUNS
from modelling.utils.io import sha256

def calibration_audit_rows(run, models, horizons=(6, 24)):
    rows = []
    for h in horizons:
        for year in range(2021, 2026):
            for config in ["market_matched", "market_onchain"]:
                for model in models:
                    folder = evidence_dir(run) / "fits" / f"{model}_{config}_h{h}_{year}"
                    spec = json.loads((folder / "fit.json").read_text())
                    stats = spec["validation"]
                    rows.append(dict(model=model, config=config, horizon_hours=h, test_year=year,
                                     calibration_year=year-1, threshold=spec["threshold"],
                                     calibration_coin_months=stats["coin_months"],
                                     calibration_false_alarms=stats["false_alarms"],
                                     calibration_budget=stats["coin_months"],
                                     final_train_rows=spec["final_train_rows"]))
    return rows



def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out-dir', type=Path,
                        default=Path('reports/validation_checks'))
    parser.add_argument('--include-h1', action='store_true',
                        help='include supplementary h1 alongside primary h6/h24')
    args = parser.parse_args()
    rows, sources = [], []
    for run in RUNS if args.include_h1 else RUNS[:2]:
        info = json.loads((run / 'run.json').read_text())
        if info.get('status') != 'complete' or not info.get('refit_after_calibration'):
            raise ValueError(f'Not a completed final refit: {run}')
        rows.extend(calibration_audit_rows(run, info['models'], info['horizons']))
        sources.append(dict(path=str(run), run_id=info['run_id'],
                            sha256=sha256(run / 'run.json')))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.out_dir / 'calibration_audit.csv', index=False)
    (args.out_dir / 'calibration_audit_manifest.json').write_text(
        json.dumps(dict(sources=sources), indent=2) + '\n')
    print(f'Saved {len(rows)} calibration audit rows.')


if __name__ == '__main__':
    main()
