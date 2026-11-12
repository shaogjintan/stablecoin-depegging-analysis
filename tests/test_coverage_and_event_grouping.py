"""Check coverage denominators and descriptive episode grouping."""
import unittest
import contextlib
import io
import json
from pathlib import Path
import tempfile
from unittest.mock import patch
import pandas as pd
from modelling.reporting.coverage_table import coverage, RAW_ONCHAIN
from data_processing.feature_schema import MARKET_FEATURES, ONCHAIN_FEATURES
from modelling.reporting.event_types import classify
from modelling.reporting.event_annotations import annotate
from modelling.reporting.event_table import group_events
from modelling.reporting import event_table
from modelling.utils.io import sha256

class ReportingTests(unittest.TestCase):

    """Check coverage denominators and descriptive event grouping."""
    def test_coverage_keeps_missing_edges_in_denominator(self):
        """Check missing edge hours remain in the coverage denominator."""
        d = pd.DataFrame(dict(coin=['USDT'] * 3, hour=pd.date_range('2021-01-01', periods=3, freq='h', tz='UTC'), trading_volume=[None, 1, None], common_sample=[False, True, False], **{c: [None, 2, None] for c in RAW_ONCHAIN}))
        for column in MARKET_FEATURES + ONCHAIN_FEATURES:
            d[column] = [None, 0., None]
        # Stored historical flags must not control current coverage.
        d["common_sample"] = False
        result = coverage(d)
        self.assertTrue(result.expected_hours.eq(3).all())
        self.assertTrue(result.observed_hours.eq(1).all())
        self.assertTrue(result.coverage_pct.between(33, 34).all())

    def test_structural_types_and_context_remain_separate(self):
        """Check event types remain separate from historical context."""
        d = pd.DataFrame(dict(coin=['USDC', 'DAI'], episode_id=['a', 'b'], start=pd.to_datetime(['2023-03-11T01:00Z', '2023-03-11T03:00Z']), n_hours=[4, 5], min_price=[0.97, 0.98]))
        eps = annotate(group_events(d, 48))
        events = eps[['event_id']].drop_duplicates()
        self.assertEqual(classify(eps, events).iloc[0]['type'], 'market-wide, staggered')
        self.assertTrue(eps.historical_context.eq('SVB bank run (Mar 2023)').all())

    def test_grouping_does_not_chain_onsets(self):
        """Check event grouping remains anchored to the first onset."""
        d = pd.DataFrame(dict(coin=['USDC', 'DAI', 'TUSD'], start=pd.date_range('2021-01-01', periods=3, freq='30h', tz='UTC')))
        self.assertEqual(group_events(d, 48).event_id.nunique(), 2)

    def test_export_regenerates_annotated_episodes_in_active_data_root(self):
        """Regenerate episode annotations from the same event report types."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            processed = root / "data/processed"
            processed.mkdir(parents=True)
            source = pd.DataFrame(dict(
                coin=["USDC", "DAI"], episode_id=["a", "b"],
                start=pd.to_datetime(["2023-03-11T01:00Z",
                                      "2023-03-11T03:00Z"]),
                end=pd.to_datetime(["2023-03-11T05:00Z",
                                    "2023-03-11T08:00Z"]),
                n_hours=[4, 5], min_price=[0.97, 0.98],
            ))
            source_path = processed / "depeg_episodes.parquet"
            source.to_parquet(source_path, index=False)
            source_hash = sha256(source_path)
            output = processed / "depeg_episodes_w_event_type.parquet"
            pd.DataFrame({"stale": [True]}).to_parquet(output)
            reports = root / "reports/depeg_events"
            with patch.object(event_table, "PROCESSED_DATA_DIR", processed), \
                    contextlib.redirect_stdout(io.StringIO()):
                event_table.main(["--out-dir", str(reports), "--no-plots"])
            annotated = pd.read_parquet(output)
            pd.testing.assert_frame_equal(annotated[source.columns], source)
            self.assertTrue(annotated.historical_context.eq(
                "SVB bank run (Mar 2023)").all())
            types = pd.read_csv(reports / "event_classification.csv")
            expected = annotated.event_id.map(types.set_index("event_id").type)
            pd.testing.assert_series_equal(
                annotated.event_type, expected, check_names=False)
            mapping = pd.read_csv(reports / "episode_event_mapping.csv")
            self.assertEqual(annotated.event_id.tolist(),
                             mapping.event_id.tolist())
            manifest = json.loads((reports / "manifest.json").read_text())
            self.assertEqual(manifest["annotated_episodes"]["sha256"],
                             sha256(output))
            self.assertEqual(sha256(source_path), source_hash)
