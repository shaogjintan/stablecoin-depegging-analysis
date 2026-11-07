"""Offline checks for processing imports and output-directory handling."""

import contextlib
import importlib
import io
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock
import zipfile

import pandas as pd

from data_processing import build_market_features
from data_processing import build_onchain_features


class ProcessingTests(unittest.TestCase):
    """Check offline imports and processing output writers."""
    def test_package_imports_have_no_filesystem_or_network_side_effects(self):
        """Check imports leave the filesystem and network untouched."""
        root = Path(__file__).resolve().parents[1]
        code = """
import importlib
import socket
from unittest.mock import patch
with patch.object(socket.socket, 'connect', side_effect=AssertionError('network')):
    for name in ['config', 'fetch_stablecoin_data', 'build_market_features',
                 'build_onchain_features', 'merge_and_label', 'splits',
                 'build_dataset']:
        importlib.import_module('data_processing.' + name)
from data_processing import fetch_stablecoin_data
assert fetch_stablecoin_data.SESSION is None
"""
        with tempfile.TemporaryDirectory() as temporary:
            environment = dict(os.environ, PYTHONPATH=str(root))
            subprocess.run(
                [sys.executable, "-c", code], cwd=temporary,
                env=environment, check=True, capture_output=True,
            )
            self.assertEqual(list(Path(temporary).iterdir()), [])

    def test_market_writer_creates_missing_output_directory(self):
        """Check the market writer creates a missing output directory."""
        frame = pd.DataFrame({
            "coin": "USDT",
            "hour": pd.date_range("2021-01-01", periods=30, freq="h", tz="UTC"),
            "price": [1.0] * 30,
            "volume": [100.0] * 30,
            "venue": "example",
            "hours_since_venue_start": range(1, 31),
        })
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "hourly.parquet"
            output = Path(temporary) / "new" / "interim" / "market.parquet"
            frame.to_parquet(source, index=False)
            with mock.patch.object(build_market_features, "HOURLY_MARKET_PATH", source), \
                 mock.patch.object(build_market_features, "OUT_PATH", output), \
                 contextlib.redirect_stdout(io.StringIO()):
                build_market_features.main()
            saved = pd.read_parquet(output)
            self.assertEqual(len(saved), len(frame))
            self.assertFalse(saved.duplicated(["coin", "hour"]).any())
            self.assertIn("volume_pct_change_24h", saved)

    def test_onchain_writer_creates_missing_output_directory(self):
        """Check the on-chain writer creates a missing output directory."""
        csv = (
            "hour,tx_count,total_volume,active_senders,active_receivers\n"
            "2021-01-01 00:00:00.000 UTC,1,2,1,1\n"
            "2021-01-01 01:00:00.000 UTC,2,3,1,2\n"
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "exports.zip"
            output = root / "new" / "interim" / "onchain.parquet"
            with zipfile.ZipFile(source, "w") as archive:
                archive.writestr("dune_onchain/usdt.csv", csv)
            with mock.patch.object(build_onchain_features, "DUNE_ONCHAIN_ZIP", source), \
                 mock.patch.object(build_onchain_features, "DUNE_ONCHAIN_DIR", root / "raw"), \
                 mock.patch.object(build_onchain_features, "OUT_PATH", output), \
                 contextlib.redirect_stdout(io.StringIO()):
                build_onchain_features.main()
            saved = pd.read_parquet(output)
            self.assertEqual(len(saved), 2)
            self.assertEqual(set(saved.coin), {"USDT"})
            self.assertIn("tx_count_logdiff_1h", saved)

    def test_merge_and_label_delegates_to_current_builder(self):
        """Check the merge entry point delegates to the canonical builder."""
        from unittest.mock import patch
        merge = importlib.import_module("data_processing.merge_and_label")
        with patch("data_processing.build_dataset.main", return_value=0) as build:
            result = merge.main()
        build.assert_called_once()
        self.assertIsNone(result)
