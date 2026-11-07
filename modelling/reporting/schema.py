"""Consistent headline metrics and supporting counts for comparison reports."""
from pathlib import Path

HEADLINE_METRICS = [
    'median_lead_hours', 'episode_recall', 'false_alarms',
    'false_alarms_per_coin_month', 'average_precision',
]
SUPPORTING_COUNTS = [
    'episodes_caught', 'targetable_episodes', 'coin_months',
    'test_rows', 'positive_hours', 'alarm_events', 'true_alarms',
]


def write_comparison(frame, output, identifiers):
    """Write five headline metrics and counts; preserve other fields separately."""
    missing = set(identifiers + HEADLINE_METRICS) - set(frame.columns)
    if missing:
        raise ValueError(f'Missing comparison columns: {sorted(missing)}')
    if frame.duplicated(identifiers).any():
        raise ValueError('Duplicate comparison identifiers')
    aligned = frame.copy()
    # Some saved yearly runs use both old and canonical count names.
    for old, new in [('rows', 'test_rows'), ('observed_coin_months', 'coin_months')]:
        if old in aligned:
            if new in aligned:
                both = aligned[old].notna() & aligned[new].notna()
                if not aligned.loc[both, old].eq(aligned.loc[both, new]).all():
                    raise ValueError(f'Conflicting count aliases: {old}, {new}')
                aligned[new] = aligned[new].combine_first(aligned[old])
            else:
                aligned[new] = aligned[old]
    columns = identifiers + HEADLINE_METRICS + [
        name for name in SUPPORTING_COUNTS if name in aligned
    ]
    extra = [name for name in frame.columns if name not in columns]
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if extra:
        frame[identifiers + extra].to_csv(
            output.with_name(output.stem + '_diagnostics.csv'), index=False,
        )
    result = aligned[columns]
    result.to_csv(output, index=False)
    return result
