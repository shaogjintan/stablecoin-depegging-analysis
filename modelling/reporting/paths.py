"""Resolve relocated saved fits without modifying original run manifests."""
from functools import lru_cache
import json
from pathlib import Path


@lru_cache(maxsize=None)
def evidence_dir(run):
    """Find the unique fit directory matching a run's recorded identity."""
    run = Path(run)
    run_id = json.loads((run / 'run.json').read_text())['run_id']
    candidates = [run] if (run / 'fits').is_dir() else []
    if not candidates:
        candidates = [p.parent for p in run.parent.parent.glob('*/*/fits')]
    matches = []
    for candidate in candidates:
        specs = sorted((candidate / 'fits').glob('*/fit.json'))
        if specs and json.loads(specs[0].read_text()).get('run_id') == run_id:
            matches.append(candidate)
    if len(matches) != 1:
        raise ValueError(f'Expected one saved evidence directory for {run_id}: {matches}')
    return matches[0]
