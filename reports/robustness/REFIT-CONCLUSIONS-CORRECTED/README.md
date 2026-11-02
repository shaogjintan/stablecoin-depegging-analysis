# Interpreting this robustness check

Reference: downward 1% for 2h. Alternatives: the specified bands and durations.

- metrics.csv: defined and targetable episodes, recall, lead, false alarms and alarm precision.
- recall_leaders.csv: highest-recall model(s) by tier/horizon; changed_from_reference marks changes. A recall leader can raise more false alarms, so inspect metrics.csv alongside it.
- onchain_effects.csv: tier2_dominates = at least as many episodes caught and no more false alarms, with one strict improvement; tier1_dominates is the reverse; tradeoff means neither dominates. Compare each alternative with the reference using changed_from_reference.

Stable results across definitions support the conclusion within this sample. Changes indicate definition sensitivity; do not count them as independent statistical replications.

Models, prediction eligibility, thresholds and alarm times are frozen. This evaluates how the existing standard-definition system performs against alternative outcomes. It does not establish performance for systems retrained and recalibrated for those definitions. AP is not recomputed against unchanged labels. Median lead is conditional on caught episodes.
