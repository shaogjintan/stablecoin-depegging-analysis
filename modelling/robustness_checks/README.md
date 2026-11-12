# Robustness checks

These modules read saved final-model scores; they do not fit new primary models.

| Module | Question | What stays fixed? |
|---|---|---|
| alternative_definitions | Does evaluation change when depeg bands/durations change? | Origins, models, cutoffs and alarm times |
| alarm_smoothing | Does requiring consecutive warnings change alarm performance? | Models, origins and original cutoffs |
| or_combination | Does combining two models' warnings change performance? | Models, origins and each component's original cutoff |

Run the alarm-rule checks from the repository root:

```sh
python -m modelling.robustness_checks.alarm_smoothing
python -m modelling.robustness_checks.or_combination
```

Defaults include all three final models, both matched feature tiers, h6/h24
and 2021–2025. Add `--horizons 1 6 24` to include supplementary h1. Repeated
`--run-dir` arguments select new final runs. `--years 2024 --horizons 6` selects
a smaller scope for inspection. `--out-dir` selects a separate report folder.
The full reproduction workflow runs these checks on its newly generated run.

## Smoothing

The default requires two consecutive same-coin calendar hours above the fixed
saved cutoff. `--hours` changes the required run length. A missing hour or the
start of a saved partition breaks confirmation. Eligible origins and metric
exposure are unchanged; the first unconfirmed hour is not removed from them.
After confirmation, apply the standard horizon-length cooldown. Calibration
uses saved practice-model scores; tests use saved final-model scores.

## OR-combination

For every pair of final models within the same tier, horizon and year, combine
raw above-cutoff flags using logical OR and then apply one shared cooldown.
Both component baselines and the OR rule use identical observations and labels;
mismatched populations fail rather than being silently intersected. This is
not a union of already cooldown-grouped alarm lists. Pairs are not selected
by test performance. Original component cutoffs are never tuned on test data.

OR is not constrained to the original calibration budget: each component's
budget does not establish a joint budget. Calibration false-alarm rates and a
budget indicator are reported for every variant. Cooldown interactions can
also move alarm times, so combined recall need not dominate both components.
Compare recall, false alarms and lead time jointly, not recall alone.

## Outputs and interpretation

Separate outputs are written to reports/robustness/alarm_smoothing/ and
reports/robustness/or_combination/. Each has by_year.csv, summary.csv,
alarm_events.csv, manifest.json and a README. The manifest records source run,
fit, partition and episode hashes. Source evidence is not rewritten.

Alarm-rule changes do not automatically create new probability scores.
Smoothing AP is calculated from the unchanged underlying model scores, so it
remains a ranking measure of that model, not a smoothing effect. Binary OR
has no unique continuous probability ranking; its AP is blank. Component
baseline AP uses each original score. In the exported alarms, rule_signal is
a binary flag and rule_threshold is 0.5, not a fitted risk-score cutoff; original
component cutoffs are recorded separately.
