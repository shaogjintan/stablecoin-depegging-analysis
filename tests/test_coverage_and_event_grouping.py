"""Check coverage denominators and descriptive episode grouping."""
import unittest
import pandas as pd
from modelling.reporting.coverage_table import coverage, RAW_ONCHAIN
from modelling.reporting.event_types import classify
from modelling.reporting.event_annotations import annotate
from modelling.reporting.event_table import group_events

class ReportingTests(unittest.TestCase):

    """Check coverage denominators and descriptive event grouping."""
    def test_coverage_keeps_missing_edges_in_denominator(self):
        """Check missing edge hours remain in the coverage denominator."""
        d = pd.DataFrame(dict(coin=['USDT'] * 3, hour=pd.date_range('2021-01-01', periods=3, freq='h', tz='UTC'), trading_volume=[None, 1, None], common_sample=[False, True, False], **{c: [None, 2, None] for c in RAW_ONCHAIN}))
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
