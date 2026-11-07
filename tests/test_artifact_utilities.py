"""Evidence resolution failure cases and streamed checksum equivalence."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from modelling.utils.artifacts import evidence_dir
from modelling.utils.io import sha256, save_json


class ArtifactUtilitiesTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / 'runs'
        self.run = self.root / 'primary' / 'metadata'
        self.run.mkdir(parents=True)
        save_json(self.run / 'run.json', {'run_id': 'expected'})
        self.relocated = self.root / 'h6_and_h24' / 'evidence'

    def make_evidence(self, path, identities):
        for number, identity in enumerate(identities):
            folder = path / 'fits' / f'fit{number}'
            folder.mkdir(parents=True)
            save_json(folder / 'fit.json', {'run_id': identity})
        (path / 'episodes.parquet').write_bytes(b'fixture')

    def test_relocated_evidence_and_metadata_changes_are_detected(self):
        self.make_evidence(self.relocated, ['expected', 'expected'])
        self.assertEqual(evidence_dir(self.run), self.relocated)
        save_json(self.relocated / 'fits' / 'fit1' / 'fit.json', {'run_id': 'other'})
        with self.assertRaisesRegex(ValueError, 'Mixed run'):
            evidence_dir(self.run)

    def test_matching_run_after_a_wrong_first_fit_is_also_rejected(self):
        self.make_evidence(self.relocated, ['other', 'expected'])
        with self.assertRaisesRegex(ValueError, 'Mixed run'):
            evidence_dir(self.run)

    def test_local_evidence_cannot_hide_a_duplicate_location(self):
        self.make_evidence(self.run, ['expected'])
        self.make_evidence(self.relocated, ['expected'])
        with self.assertRaisesRegex(ValueError, 'Expected one'):
            evidence_dir(self.run)

    def test_incomplete_metadata_is_rejected(self):
        self.make_evidence(self.relocated, ['expected'])
        (self.relocated / 'fits' / 'unfinished').mkdir()
        with self.assertRaisesRegex(ValueError, 'Incomplete fit metadata'):
            evidence_dir(self.run)

    def test_missing_episode_evidence_is_rejected(self):
        folder = self.relocated / 'fits' / 'fit0'
        folder.mkdir(parents=True)
        save_json(folder / 'fit.json', {'run_id': 'expected'})
        with self.assertRaisesRegex(ValueError, 'Missing episode evidence'):
            evidence_dir(self.run)

    def test_empty_and_multi_chunk_hashes_match_without_read_bytes(self):
        for payload in [b'', bytes(range(256)) * 10001]:
            path = Path(self.temp.name) / 'hash_input'
            path.write_bytes(payload)
            with patch.object(Path, 'read_bytes', side_effect=AssertionError('unbounded read')):
                self.assertEqual(sha256(path), hashlib.sha256(payload).hexdigest())

    def test_json_serialization_remains_unchanged(self):
        path = Path(self.temp.name) / 'metadata.json'
        value = {'path': Path('relative/path'), 'count': 2}
        save_json(path, value)
        self.assertEqual(path.read_text(), json.dumps(value, indent=2, default=str))
