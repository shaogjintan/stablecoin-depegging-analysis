"""Export saved-model interpretation tables and figures.

Export coefficients and importance from saved final pipelines; no training.
"""
import argparse
import json
from pathlib import Path
from modelling.utils.artifacts import evidence_dir, FINAL_RUNS
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from modelling.utils.io import sha256
RUNS = FINAL_RUNS[:2]


def shap_ranking(runs, horizon):
    """Pool absolute raw-margin SHAP over the selected runs' test rows.

    Args:
        runs: Completed run directories containing final XGBoost fits.
        horizon: Forecast horizon to interpret for the on-chain tier.

    Returns:
        A feature ranking and provenance for every contributing partition.
    """
    from xgboost import DMatrix

    totals, features, count, sources = None, None, 0, []
    for run in runs:
        for folder in sorted((evidence_dir(run) / 'fits').iterdir()):
            spec = json.loads((folder / 'fit.json').read_text())
            if (spec['model'] != 'xgboost' or spec['horizon'] != horizon
                    or spec['config'] != 'market_onchain'):
                continue
            current = spec['features']
            if features is not None and features != current:
                raise ValueError('Inconsistent XGBoost feature order')
            features = current
            path = folder / 'pipeline.joblib'
            if sha256(path) != spec['pipeline_sha256']:
                raise ValueError(f'Pipeline changed: {folder}')
            bundle = joblib.load(path)
            test_path = folder / 'test.parquet'
            X = pd.read_parquet(test_path, columns=features)
            if bundle['imputer'] is not None:
                X = bundle['imputer'].transform(X)
            values = bundle['model'].clf_.get_booster().predict(
                DMatrix(X, nthread=1), pred_contribs=True
            )[:, :-1]
            contribution = np.abs(values).sum(axis=0, dtype=np.float64)
            totals = contribution if totals is None else totals + contribution
            count += len(X)
            sources.append(dict(path=str(test_path), sha256=sha256(test_path),
                                pipeline_sha256=spec['pipeline_sha256']))
    if not count:
        raise ValueError(f'No XGBoost test rows for h={horizon}')
    ranking = pd.DataFrame(dict(feature=features, mean_abs_shap=totals / count))
    return ranking.sort_values('mean_abs_shap', ascending=False), sources


def main():
    """Export saved-model interpretation tables and figures."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out-dir', type=Path, default=Path('reports/model_interpretation'))
    p.add_argument('--run-dir', type=Path, action='append')
    p.add_argument('--recompute-shap', action='store_true',
                   help='calculate SHAP from selected runs, not saved rankings')
    args = p.parse_args()
    runs = args.run_dir or RUNS
    if args.run_dir and not args.recompute_shap:
        p.error('--run-dir requires --recompute-shap to avoid old rankings')
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for name in ['log_reg', 'random_forest', 'xgboost']:
        (args.out_dir / name).mkdir(exist_ok=True)
    coefficients = []
    importance = []
    sources = []
    for run in runs:
        manifest = json.loads((run / 'run.json').read_text())
        if manifest.get('status') != 'complete' or not manifest.get('refit_after_calibration'):
            raise ValueError(f'Not a completed final refit: {run}')
        sources.append(dict(path=str(run), run_id=manifest['run_id'], manifest_sha256=sha256(run / 'run.json')))
        for folder in sorted((evidence_dir(run) / 'fits').iterdir()):
            spec = json.loads((folder / 'fit.json').read_text())
            path = folder / 'pipeline.joblib'
            if sha256(path) != spec['pipeline_sha256']:
                raise ValueError(f'Pipeline changed: {folder}')
            model = joblib.load(path)['model']
            common = dict(run_id=manifest['run_id'], model=spec['model'], config=spec['config'], horizon_hours=spec['horizon'], test_year=spec['year'])
            if spec['model'] == 'logreg':
                coefficients.extend((dict(**common, feature=f, coefficient=float(v), selected_C=spec['selected_C']) for f, v in zip(model.cols_, model.clf_.coef_[0])))
            elif spec['model'] == 'random_forest':
                importance.extend((dict(**common, feature=f, importance=float(v)) for f, v in zip(spec['features'], model.feature_importances_)))
    pd.DataFrame(coefficients).to_csv(args.out_dir / 'log_reg' / 'lr_coefficients.csv', index=False)
    rf = pd.DataFrame(importance)
    rf.to_csv(args.out_dir / 'random_forest' / 'rf_importance_by_year.csv', index=False)
    for h in [6, 24]:
        for model, frame, value in [('logreg', pd.DataFrame(coefficients), 'coefficient'), ('random_forest', rf, 'importance')]:
            subset = frame.loc[frame.horizon_hours.eq(h) & frame.config.eq('market_onchain')]
            values = subset.groupby('feature')[value].mean()
            if model == 'logreg':
                values = values.loc[values.abs().nlargest(12).index].sort_values()
            else:
                values = values.nlargest(12).sort_values()
            fig, ax = plt.subplots(figsize=(8, 5))
            values.plot.barh(ax=ax)
            ax.set_title(f'Final {model}, market + on-chain, h={h}h')
            ax.set_xlabel('Mean standardised coefficient across years' if model == 'logreg' else 'Mean impurity importance across years')
            fig.tight_layout()
            fig.savefig(args.out_dir / ('log_reg' if model == 'logreg' else model) / f'{model}_h{h}.png', dpi=160)
            plt.close(fig)
        filename = f'shap_ranking_xgboost_t2_h{h}.csv'
        if args.recompute_shap:
            ranking, evidence = shap_ranking(runs, h)
            sources.extend(evidence)
        else:
            source = runs[0] / filename
            ranking = pd.read_csv(source)
            sources.append(dict(path=str(source), sha256=sha256(source)))
        values = ranking.set_index('feature').mean_abs_shap.nlargest(12).sort_values()
        ranking.to_csv(args.out_dir / 'xgboost' / filename, index=False)
        fig, ax = plt.subplots(figsize=(8, 5))
        values.plot.barh(ax=ax)
        ax.set_title(f'Final XGBoost, market + on-chain, h={h}h')
        ax.set_xlabel('Mean absolute raw-margin SHAP over saved test rows')
        fig.tight_layout()
        fig.savefig(args.out_dir / 'xgboost' / f'xgboost_shap_h{h}.png', dpi=160)
        plt.close(fig)
    (args.out_dir / 'manifest.json').write_text(json.dumps(dict(sources=sources, notes=['LR coefficients follow training winsorisation and standardisation, with derived features.', 'RF impurity importance and SHAP are descriptive, not causal.', 'XGB SHAP recalculated from selected test partitions.' if args.recompute_shap else 'XGB SHAP rankings reused from saved final-model analysis.']), indent=2) + '\n')
    print('Saved final-model interpretation:', args.out_dir)
if __name__ == '__main__':
    main()
