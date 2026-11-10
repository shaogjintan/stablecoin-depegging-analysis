"""Check fresh-run SHAP rankings use saved model and test evidence."""

import json
from pathlib import Path
import tempfile
import unittest

import joblib
import numpy as np
import pandas as pd
from xgboost import DMatrix

from modelling.models.xgboost_model import XGBModel
from modelling.reporting.model_interpretation import shap_ranking
from modelling.utils.io import sha256


class InterpretationShapTests(unittest.TestCase):
    """Validate native XGBoost contributions against a tiny saved pipeline."""

    def test_ranking_matches_native_test_contributions(self):
        """Recalculate SHAP from the selected model and saved test features."""
        X = pd.DataFrame({'a': [0., 1., 2., 3.], 'b': [2., 1., 2., 1.]})
        model = XGBModel(n_estimators=2, n_jobs=1, max_depth=1)
        model.fit(X, [0, 0, 1, 1])
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary) / 'primary' / 'fresh'
            folder = run / 'fits' / 'xgboost_market_onchain_h6_2024'
            folder.mkdir(parents=True)
            (run / 'run.json').write_text(json.dumps({'run_id': 'fresh'}))
            pd.DataFrame().to_parquet(run / 'episodes.parquet')
            X.to_parquet(folder / 'test.parquet', index=False)
            path = folder / 'pipeline.joblib'
            joblib.dump(dict(model=model, imputer=None), path)
            (folder / 'fit.json').write_text(json.dumps(dict(
                run_id='fresh', model='xgboost', horizon=6,
                config='market_onchain', features=list(X),
                pipeline_sha256=sha256(path),
            )))
            ranking, sources = shap_ranking([run], 6)
            expected = np.abs(model.clf_.get_booster().predict(
                DMatrix(X), pred_contribs=True
            )[:, :-1]).mean(axis=0)
            np.testing.assert_allclose(
                ranking.set_index('feature').loc[list(X), 'mean_abs_shap'],
                expected,
            )
            self.assertEqual(sources[0]['sha256'], sha256(folder / 'test.parquet'))
            with self.assertRaisesRegex(ValueError, 'No XGBoost test rows'):
                shap_ranking([run], 24)
