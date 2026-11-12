# Alarm-rule robustness

Report every available model pair within the same tier, horizon and test year. Components must have identical origins and labels.

An hour is flagged if either model reaches its own fixed saved calibration cutoff. Apply one standard cooldown to the union of flags.

No optimisation of thresholds, pairs or alarm budgets uses test outcomes. This is not the historical test-optimised OR analysis.

The OR rule may exceed the original calibration budget. Inspect calibration and test false-alarm rates alongside recall and lead time. Cooldown interactions mean OR recall need not dominate both components.

OR combines binary decisions, not probabilities; its average precision is undefined. Component AP uses each original model score.

by_year.csv includes calibration and test metrics. summary.csv pools test years. AP for smoothing refers to the unchanged underlying model scores, not a newly fitted or smoothed probability. OR has no unique probability ranking, so its AP is blank. Compare recall, false alarms and lead time jointly; no improvement is guaranteed.
