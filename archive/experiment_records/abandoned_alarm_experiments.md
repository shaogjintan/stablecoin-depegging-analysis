# Abandoned alarm experiments

These experiments are not part of the final protocol. Original code is
preserved in delete/archive_cleanup/history/legacy_modelling/:

- alarm_smoothing.py: require consecutive above-cutoff scores before alarming.
- alarm_cascade.py: reuse h24 model scores for other forecast horizons.
- rolling_threshold_comparison.py: compare earlier rolling-cutoff strategies.
- partition_report.py: report folds through the retired model-preparation API.

None has a caller in maintained code. Their imports and launch commands refer
to retired modules. They are distinct exploratory ideas, not exact duplicates
of current scripts. No conclusions or performance claims are established by
retaining this note. Final cutoffs and alarms use the maintained training and
evaluation packages; final fold reporting uses current sample preparation.
