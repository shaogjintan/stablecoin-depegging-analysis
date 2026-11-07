"""Resolve relocated saved fits without modifying original run manifests."""
import json
from pathlib import Path

FINAL_RUNS = [
    Path('artifacts') / horizons / models
    for horizons, models in [
        ('h6_and_h24', 'random_forest_xgboost'),
        ('h6_and_h24', 'logistic_regression'),
        ('h1', 'random_forest_xgboost'),
        ('h1', 'logistic_regression'),
    ]
]


def evidence_dir(run):
    """Locate one evidence directory whose fit metadata all belongs to this run.

    Search the supplied location and the current two-level run layout. Inspect
    every fit rather than trusting the first one. Results are not cached, so
    a move or metadata change in the same process cannot reuse a stale result.
    This resolves identity and location; the verifier checks fit-cell scope.
    """
    run = Path(run)
    run_id = json.loads((run / 'run.json').read_text())['run_id']
    candidates = [run, *(
        path.parent for path in run.parent.parent.glob('*/*/fits')
    )]
    matches, seen = [], set()
    for candidate in candidates:
        physical = candidate.resolve()
        if physical in seen or not (candidate / 'fits').is_dir():
            continue
        seen.add(physical)
        folders = sorted(path for path in (candidate / 'fits').iterdir()
                         if path.is_dir())
        identities, problems = [], []
        for folder in folders:
            path = folder / 'fit.json'
            try:
                identities.append(json.loads(path.read_text())['run_id'])
            except (OSError, ValueError, KeyError, TypeError) as exc:
                problems.append(f'{path}: {exc}')
        if run_id not in identities:
            continue
        if problems:
            raise ValueError(f'Incomplete fit metadata for {run_id}: {problems}')
        if set(identities) != {run_id}:
            raise ValueError(f'Mixed run identities in {candidate / "fits"}')
        if not (candidate / 'episodes.parquet').is_file():
            raise ValueError(f'Missing episode evidence: {candidate}')
        matches.append(candidate)
    if len(matches) != 1:
        raise ValueError(f'Expected one saved evidence directory for {run_id}: {matches}')
    return matches[0]
