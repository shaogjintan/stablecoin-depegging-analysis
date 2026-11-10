# Exhaustive lowest-feasible RF and XGBoost

Python 3.12 pinned environment. Frozen corrected source: runs/XGB-NEW.
Tier 1 is refitted on Tier 2 eligible rows for every partition. Horizons 1, 6, 24; test years 2021-2025. RF uses the shared 500-tree, depth-8, minimum-leaf-50 configuration. XGBoost reuses fixed 300-tree scores. All thresholds are selected on inner_val, with budget 1 FA per observed coin-month, without test-label selection or refitting after calibration.

summary.csv contains exhaustive_lowest only. summary_all_policies.csv adds the maximum-recall/fewer-FA reference policy using identical scores. comparison_by_tier.csv is the paired table. by_year.csv records thresholds and validation/test counts. alarm_events.csv records individual alarms. Saved RF pipelines and both models' scored rows are under fits/.

Validation checks passed: 120 policy-fold rows from 60 model evaluations; matched test counts/episode denominators; all validation budgets; exact-policy XGB totals reproduce XGB-NEW. Existing production final-results tables were not overwritten.

On-chain features reduce realised false alarms at every horizon for both models. RF catches the same 4/33 and 6/34 episodes at h6/h24 with fewer false alarms; it loses its one h1 catch. XGBoost catches the same 4/34 episodes at h24 with fewer false alarms, but loses one h6 catch. These are descriptive results on inspected test years, not proof of a general improvement.
