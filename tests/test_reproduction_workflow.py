"""Check isolated reproduction paths, execution order, and failure reporting."""

import contextlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from reports import reproduce


class ReproductionWorkflowTests(unittest.TestCase):
    """Keep fresh reproduction separate from historical run evidence."""

    def test_full_plan_uses_only_the_new_run(self):
        """All run consumers must receive the new run explicitly."""
        output = Path('/tmp/reproduction-plan-test').resolve()
        plan = reproduce.full_plan(output)
        modules = [module for module, _ in plan]
        self.assertLess(modules.index('data_processing.build_dataset'),
                        modules.index('modelling.training.train_models'))
        run = str(output / 'artifacts/runs/primary/ALL-MODELS')
        for module, args in plan:
            if module in {
                'modelling.validation.verify_model_results',
                'modelling.robustness_checks.alternative_definitions',
                'modelling.robustness_checks.alarm_smoothing',
                'modelling.robustness_checks.or_combination',
                'modelling.reporting.model_results',
                'modelling.reporting.calibration_audit',
                'modelling.reporting.detection_by_type',
                'modelling.reporting.benchmarks',
                'modelling.reporting.model_interpretation',
            }:
                self.assertEqual(args[args.index('--run-dir') + 1], run)
        self.assertNotIn('REFIT-LOWEST', json.dumps(plan))
        self.assertIn('--recompute-shap', plan[-1][1])

    def test_failure_stops_the_workflow_and_records_progress(self):
        """Never run reporting after an upstream failure or overwrite a run."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            hourly = root / 'data/raw/hourly_stablecoin.parquet'
            hourly.parent.mkdir(parents=True)
            hourly.write_bytes(b'retained source')
            (root / 'data/raw/dune_onchain.zip').write_bytes(b'retained archive')
            output = root / 'fresh'
            plan = [('first', []), ('second', []), ('third', [])]
            env = dict(os.environ, STABLECOIN_DATA_DIR=str(output / 'data'))
            failure = subprocess.CalledProcessError(1, 'second')
            with patch.object(reproduce, 'ROOT', root), \
                    patch.object(reproduce.subprocess, 'run',
                                 side_effect=[None, failure]) as run, \
                    contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaises(subprocess.CalledProcessError):
                    reproduce.run_full(output, plan, env)
                self.assertEqual(run.call_count, 2)
                state = json.loads((output / 'reproduction.json').read_text())
                self.assertEqual(state['status'], 'failed')
                self.assertEqual(state['completed'], ['first'])
                self.assertEqual(state['current'], 'second')
                with self.assertRaises(FileExistsError):
                    reproduce.run_full(output, plan, env)
            self.assertEqual(hourly.read_bytes(), b'retained source')
            self.assertEqual(
                (output / 'data/raw/hourly_stablecoin.parquet').read_bytes(),
                b'retained source',
            )

    def test_processed_mode_copies_frozen_inputs_without_rebuilding(self):
        """Retraining uses exact retained bytes in an isolated workspace."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = ['raw/hourly_stablecoin.parquet', 'raw/dune_onchain.zip',
                     'processed/master.parquet', 'processed/depeg_episodes.parquet',
                     'processed/dataset_metadata.json']
            for name in paths:
                path = root / 'data' / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(name.encode())
            output = root / 'fresh'
            env = dict(os.environ, STABLECOIN_DATA_DIR=str(output / 'data'))
            plan = reproduce.full_plan(output, from_processed=True)
            self.assertEqual(plan[0][0], 'modelling.training.train_models')
            self.assertFalse(any(m.startswith('data_processing.')
                                 for m, _ in plan))
            with patch.object(reproduce, 'ROOT', root), \
                    patch.object(reproduce.subprocess, 'run'), \
                    contextlib.redirect_stdout(io.StringIO()):
                reproduce.run_full(output, plan, env, from_processed=True)
            for name in paths:
                self.assertEqual((output / 'data' / name).read_bytes(),
                                 name.encode())
            state = json.loads((output / 'reproduction.json').read_text())
            self.assertEqual(state['input_mode'], 'retained-processed')
            self.assertEqual(len(state['sources']), 5)

    def test_saved_reports_use_published_run_and_workspace(self):
        """Published report regeneration must not select superseded runs."""
        baseline = {'run_dir': 'artifacts/final_run'}
        with patch('modelling.utils.artifacts.published_baseline',
                   return_value=baseline):
            plan = reproduce.saved_plan()
        self.assertNotIn('modelling.training.train_models',
                         [module for module, _ in plan])
        run = (reproduce.ROOT / baseline['run_dir']).resolve()
        for module, args in plan:
            if '--run-dir' in args:
                self.assertEqual(args[args.index('--run-dir') + 1], str(run))
            if module == 'modelling.reporting.by_coin':
                self.assertEqual(args[args.index('--artifacts') + 1],
                                 str(run.parent.parent))
        self.assertIn('--include-h1', dict(plan)[
            'modelling.reporting.calibration_audit'])

    def test_dry_run_has_no_output_side_effects(self):
        """Print the entire workflow without acquiring data or making files."""
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / 'not-created'
            result = subprocess.run([
                sys.executable, '-m', 'reports.reproduce', '--full',
                '--dry-run', '--out-dir', str(output),
            ], cwd=reproduce.ROOT, check=True, capture_output=True, text=True)
            state = json.loads(result.stdout)
            self.assertEqual(state['mode'], 'full')
            self.assertEqual(state['dataset_root'], str(output.resolve() / 'data'))
            self.assertEqual(state['interim_root'], str(
                output.resolve() / 'data/interim_reproduction'))
            self.assertFalse(output.exists())

    def test_data_root_override_reaches_every_input_path(self):
        """Child processes resolve dataset inputs inside the selected root."""
        with tempfile.TemporaryDirectory() as temporary:
            code = (
                'from data_processing import config as c; '
                'from pathlib import Path; '
                'assert all(p.is_relative_to(c.DATA_DIR) for p in '
                '[c.HOURLY_MARKET_PATH, c.MARKET_FEATURES_PATH, '
                'c.ONCHAIN_FEATURES_PATH, c.ACTIVE_MASTER_PATH, '
                'c.ACTIVE_EPISODES_PATH, c.DUNE_ONCHAIN_DIR])'
            )
            subprocess.run([sys.executable, '-c', code], cwd=reproduce.ROOT,
                           env=dict(os.environ, STABLECOIN_DATA_DIR=temporary),
                           check=True, capture_output=True)

    def test_reproduction_interim_override_reaches_feature_builders(self):
        """Both feature paths follow the isolated interim output directory."""
        with tempfile.TemporaryDirectory() as temporary:
            interim = Path(temporary) / 'interim_reproduction'
            code = (
                'from data_processing import config as c; '
                'from pathlib import Path; '
                f'expected = Path({str(interim)!r}); '
                'assert c.MARKET_FEATURES_PATH.parent == expected; '
                'assert c.ONCHAIN_FEATURES_PATH.parent == expected'
            )
            subprocess.run(
                [sys.executable, '-c', code], cwd=reproduce.ROOT,
                env=dict(os.environ, STABLECOIN_INTERIM_DIR=str(interim)),
                check=True, capture_output=True,
            )
