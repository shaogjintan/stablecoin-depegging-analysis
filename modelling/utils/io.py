"""Checksums and JSON serialization for saved research evidence."""
import hashlib
import json
from pathlib import Path


def sha256(path):
    """Hash file bytes using bounded memory, including for large model files."""
    digest = hashlib.sha256()
    with Path(path).open('rb') as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def save_json(path, value):
    """Write JSON using the existing research-metadata serialization format."""
    Path(path).write_text(json.dumps(value, indent=2, default=str), encoding='utf-8')
