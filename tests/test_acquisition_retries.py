"""Check acquisition failure handling without making network requests."""

import contextlib
import sys
from types import SimpleNamespace
import io
import unittest
from unittest.mock import Mock, patch

from data_processing import fetch_stablecoin_data as acquisition


class AcquisitionRetryTests(unittest.TestCase):
    """Ensure request failures use retry paths rather than undefined names."""

    def test_request_errors_are_handled_for_all_three_sources(self):
        """Simulate connection failures with no network or retry delays."""
        session = Mock()
        session.get.side_effect = ConnectionError('offline test')
        with patch.dict(sys.modules, {
                'requests': SimpleNamespace(RequestException=ConnectionError)
             }), \
                patch.object(acquisition, '_get_session', return_value=session), \
                patch.object(acquisition.time, 'sleep'), \
                contextlib.redirect_stdout(io.StringIO()):
            result = acquisition.fetch_defillama_hourly(
                'USDT', 'coingecko:tether', '2021-01-01',
                '2021-01-01 01:00',
            )
            self.assertTrue(result.empty)
            self.assertEqual(session.get.call_count, 5)
            session.get.reset_mock()
            with self.assertRaisesRegex(RuntimeError, 'repeated failures'):
                acquisition.fetch_binance_month('USDTUSD', '2021-01')
            self.assertEqual(session.get.call_count, 4)
            session.get.reset_mock()
            with self.assertRaisesRegex(RuntimeError, 'repeated failures'):
                acquisition.fetch_bitfinex_range(
                    'tUSTUSD', '2021-01-01', '2021-01-01 01:00'
                )
            self.assertEqual(session.get.call_count, 5)
