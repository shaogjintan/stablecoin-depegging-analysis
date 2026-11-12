# Alarm-rule robustness

Require 2 consecutive eligible same-coin calendar hours above the fixed saved cutoff. Gaps and partition starts break runs.

Use saved practice scores for calibration and saved final-model scores for test years. No new training or cutoff selection is performed.

Apply the standard horizon-length cooldown after the confirmation rule. The denominator remains the original eligible population.

Average precision uses unchanged model scores and therefore does not measure a change caused by smoothing.

by_year.csv includes calibration and test metrics. summary.csv pools test years. AP for smoothing refers to the unchanged underlying model scores, not a newly fitted or smoothed probability. OR has no unique probability ranking, so its AP is blank. Compare recall, false alarms and lead time jointly; no improvement is guaranteed.
