"""Checks for strict on-chain inputs and conflict-safe ZIP extraction."""
import contextlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import numpy as np
import pandas as pd

from data_processing import build_onchain_features as builder


class OnchainValidationTests(unittest.TestCase):
    """Check invalid exports fail before feature output is replaced."""

    def setUp(self):
        """Create isolated source, extraction, and output locations."""
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.archive = self.root / 'source.zip'
        self.raw = self.root / 'raw'
        self.output = self.root / 'features.parquet'
        for name, value in [('DUNE_ONCHAIN_ZIP', self.archive),
                            ('DUNE_ONCHAIN_DIR', self.raw),
                            ('OUT_PATH', self.output)]:
            patcher = patch.object(builder, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)

    def make_archive(self, members):
        """Write a ZIP fixture with the supplied member names and bytes."""
        with zipfile.ZipFile(self.archive, 'w') as source:
            for name, payload in members:
                source.writestr(name, payload)

    def hourly(self):
        """Return valid hourly on-chain observations for one coin."""
        return pd.DataFrame({
            'coin': ['USDT', 'USDT'],
            'hour': pd.date_range('2021-01-01', periods=2, freq='h', tz='UTC'),
            'tx_count': [1., 2.], 'tx_volume': [3., 4.],
            'active_senders': [1., 2.], 'active_receivers': [1., 2.],
        })

    def test_identical_exports_are_reused_without_rewriting(self):
        """Check an identical export keeps its bytes, name, and timestamp."""
        self.raw.mkdir()
        existing = self.raw / 'USDT.csv'
        existing.write_bytes(b'unchanged export')
        before = existing.stat().st_mtime_ns
        self.make_archive([('dune/usdt.csv', b'unchanged export')])
        with contextlib.redirect_stdout(io.StringIO()):
            result = builder.extract_fiat_csvs()
        self.assertEqual(result['USDT'], existing)
        self.assertEqual(existing.stat().st_mtime_ns, before)
        self.assertEqual(existing.read_bytes(), b'unchanged export')
        self.assertEqual(len(list(self.raw.iterdir())), 1)

    def test_late_conflict_prevents_all_extraction_writes(self):
        """Check a later conflict cannot leave earlier new exports behind."""
        self.raw.mkdir()
        existing = self.raw / 'usdt.csv'
        existing.write_bytes(b'original')
        self.make_archive([('usdc.csv', b'new'), ('usdt.csv', b'conflict')])
        with self.assertRaisesRegex(ValueError, 'conflicts'):
            builder.extract_fiat_csvs()
        self.assertEqual(existing.read_bytes(), b'original')
        self.assertEqual([p.name for p in self.raw.iterdir()], ['usdt.csv'])

    def test_duplicate_coin_members_are_rejected_before_writing(self):
        """Check two archive members cannot ambiguously select one coin."""
        self.make_archive([('a/usdt.csv', b'a'), ('b/USDT.csv', b'b')])
        with self.assertRaisesRegex(ValueError, 'Multiple archive exports'):
            builder.extract_fiat_csvs()
        self.assertFalse(self.raw.exists())

    def test_unsafe_member_paths_are_rejected(self):
        """Check recognized coin exports cannot use traversal paths."""
        self.make_archive([('../usdt.csv', b'content')])
        with self.assertRaisesRegex(ValueError, 'Unsafe archive member'):
            builder.extract_fiat_csvs()
        self.assertFalse(self.raw.exists())

    def test_duplicate_hours_are_rejected_instead_of_dropped(self):
        """Check duplicate observations cannot silently change features."""
        frame = self.hourly()
        duplicate = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)
        with self.assertRaisesRegex(ValueError, 'duplicate coin-hour'):
            builder.build_change_features(duplicate)

    def test_invalid_keys_and_observed_values_are_rejected(self):
        """Check malformed hours and numeric levels fail validation."""
        for column, value in [('hour', pd.NaT), ('coin', None),
                              ('tx_count', -1.), ('tx_volume', np.inf),
                              ('active_senders', 'invalid')]:
            with self.subTest(column=column, value=value):
                frame = self.hourly()
                if isinstance(value, str):
                    frame[column] = frame[column].astype(object)
                frame.loc[0, column] = value
                with self.assertRaises(ValueError):
                    builder.build_change_features(frame)
        frame = self.hourly()
        frame.loc[0, 'hour'] += pd.Timedelta(minutes=30)
        with self.assertRaisesRegex(ValueError, 'aligned to UTC hours'):
            builder.build_change_features(frame)

    def test_missing_observations_remain_missing(self):
        """Check missing metric observations are allowed and never zeroed."""
        frame = self.hourly()
        frame.loc[0, 'tx_count'] = np.nan
        result = builder.build_change_features(frame)
        self.assertTrue(pd.isna(result.tx_count.iloc[0]))
        self.assertTrue(result.tx_count_logdiff_1h.isna().all())

    def test_malformed_csv_preserves_existing_feature_output(self):
        """Check an invalid export cannot replace an existing feature table."""
        self.make_archive([('usdt.csv', b'hour,tx_count\ninvalid,1\n')])
        self.output.write_bytes(b'preserved output')
        with contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(ValueError, 'missing required columns'):
                builder.main()
        self.assertEqual(self.output.read_bytes(), b'preserved output')
