# RF/XGBoost reproduction investigation

Reviewed on 2026-11-10. This review compares the retained final runs with
`artifacts/reproduction/artifacts/runs/primary/ALL-MODELS/` (run ID
`cdd872b4-68d3-4561-a81b-482998b65b1b`). No retained data, fitted models or
report outputs were replaced.

## Findings

The full reproduction completed. Its verification record passed all 90 cells,
including saved-model score replay. The review additionally checked all 180
practice/final pipeline hashes and confirmed that the generated overall and
yearly headline metrics match the reproduced run.

The rebuilt episode table matches the retained episode table exactly. Apart
from the five documented historical diagnostic columns, the master tables
match exactly except for rolling volatility, which passes the existing
absolute tolerance of 1e-9. Eligibility, labels and other modelling features
match. However, this numerical tolerance is insufficient to guarantee the
same fitted trees.

| Feature | Nonidentical rows | Maximum absolute difference | Rows different after float32 conversion |
|---|---:|---:|---:|
| realised_vol_6h | 205471 | 4.94929160515118e-10 | 350 |
| realised_vol_24h | 6751 | 8.488079171918877e-10 | 572 |

For 284 six-hour rows and 496 twenty-four-hour rows, one of the two values is
exactly zero. Small absolute differences near zero can survive the float32
conversion used in tree fitting and affect split decisions. They need not
produce small changes in the resulting fitted model.

The retained and reproduced runs record the same package versions. The
retained and current calendar-market functions also give identical volatility
when run on the same current inputs. Neither reconstructs the frozen
volatility values exactly. The precise historical numerical execution that
produced those frozen values has not been established; the investigation does
not attribute this to a confirmed package-version or formula change.

## Direct numerical check of the largest differences

A follow-up check on 2026-11-11 recomputed the two windows with the largest
absolute disagreement independently, including 70-digit decimal arithmetic
on the stored prices:

- USDC, 2023-03-26 19:00 UTC, six-hour volatility: all seven prices required
  for the six returns equal 1.001. Every return is zero. The frozen value is
  4.94929160515118e-10; the rebuilt and independent values are exactly zero.
- PAX, 2025-08-04 05:00 UTC, twenty-four-hour volatility: all 25 prices equal
  1.0. Every return is zero. The frozen value is 8.488079171918877e-10; the
  rebuilt and independent values are exactly zero.

For these windows, the nonzero frozen values are confirmed numerical
residuals rather than actual within-window price variation. This establishes
that the largest disagreements involve numerical error. It does not identify
which historical execution condition produced the residuals, or prove the
origin of every differing row. No stored data were altered by this check.

## Controlled refits

The `market_matched`, h=6, test-year 2024 fold was checked for both RF and
XGBoost. All four saved partitions have the same rows, order, labels and
feature order. The only differing model inputs are volatility values.
Estimator settings match, including the XGBoost booster configuration.

Using the current training code, XGBoost refitted on the retained training
matrix reproduces the retained booster trees and predictions exactly.
Refitting on the rebuilt matrix reproduces the new booster trees and
predictions exactly. This isolates an input-dependent fitting difference in
this fold, rather than an unexplained change in the training implementation.
The old booster also returns identical predictions on old and new test
features in this fold: its fitting inputs, rather than direct test-time
rounding, account for the model difference there.

RF refitted on retained inputs matches retained test probabilities within
2.220446049250313e-16. A refit on rebuilt inputs matches the reproduced RF
probabilities within 1.6653345369377348e-16. Parallel probability summation can
change the final
floating-point bits, so byte equality of probabilities is a stronger condition
than this check establishes.

These are controlled refits of one representative fold, not repeat training
of all 90 cells or all tuning/calibration stages.

## Effect on reported results

All six pooled LR results match within rtol=1e-10 and atol=1e-12. Episode recall
matches for all 18 pooled results and all 90 yearly cells. RF/XGBoost pooled
average precision differs in all twelve cells. Eight pooled false-alarm counts
differ (maximum absolute difference 13); two pooled median lead times differ
(maximum difference one hour). Exact reproduction of the retained reported
numbers is therefore not established by the successful full workflow audit.

## Implications

Saved-result reproduction remains the route for regenerating the original
reported evidence. Full rebuilding and retraining currently produces a valid,
audited new run but must not be described as reproducing every original number.

For a new reproducible baseline, first specify and test a numerically stable
rolling-volatility calculation, then rebuild and retrain into a new workspace
and update all dependent reports together. Alternatively, retraining on the
retained processed dataset isolates model training from feature reconstruction.
Do not silently round features or substitute frozen volatility into a claimed
raw-source rebuild: either choice must be explicit and validated. Original
artifacts should remain available for comparison.

## Corrective changes and repeat validation (2026-11-11)

No original processed inputs, fitted evidence or report tables were replaced.
The following changes address the demonstrated numerical failure modes:

- Rolling sample deviation now uses independent centered calculations with
  `math.fsum` for each complete window. No accumulated rolling state crosses
  windows; genuine small variations are retained without a numerical cutoff.
- RF probabilities are summed in fixed tree order with one prediction worker.
  The training worker count and random seed remain unchanged.
- Fresh dataset metadata records the volatility method. Rebuild comparisons
  for that method require exact feature equality. The historical tolerance is
  retained only for comparisons against earlier datasets.
- `reports.reproduce --full --from-processed` copies the frozen processed
  dataset into an isolated workspace before retraining. It records source
  hashes and skips reconstruction. This is distinct from source rebuilding.

Two independent real-data builds, including reversed source-row order, produced
exactly equal master tables. Original labels and eligibility flags remained
unchanged. The only model-feature changes versus the retained master are the
two volatility columns, still within the measured historical bounds.

For RF and XGBoost in the market-matched, h=6, year-2021 fold, both datasets
were independently used for practice fitting, cutoff calibration and final
refitting. Validation scores, selected cutoffs, test scores and grouped alarms
matched exactly across the two repetitions. Local diagnostic evidence is under
`artifacts/reproducibility_checks/stable_inputs/validation.json` (ignored in Git).

This is not a second full 90-cell training run. New source-based report tables
must not replace the retained baseline until a complete run and comparison
have been reviewed. The frozen-processed workflow is tested for isolated input
copying and stage selection; it has not yet completed a new all-90-cell run.

## Full primary training comparison

The subsequent stable-a/stable-b runs completed all 90 primary fit cells and
passed their saved model-score audits. Their 455 compared training/data/result
tables and scientific fit metadata match exactly. This supersedes the earlier
small-fold-only validation for primary training. The original report numbers
still differ from the stable baseline. See
[the stable comparison](stable_reproduction_comparison.md) for scope and all
changed pooled headline values against the original reports. Downstream report
comparison remains separate from completed primary training verification.

## Published reports updated

After completion of both full workflows and the 490-check comparison, current
`reports/` outputs were updated from stable-a. Superseded snapshots were subsequently removed. Promotion checksums remain
in `reports/validation_checks/reproducibility/report_promotion.json`.
`reports/baseline.json` selects the active run and dataset; saved-report
regeneration now uses that workspace explicitly. All 29 current report CSVs
match stable-a exactly. See the stable comparison document for the full
promotion scope, added h1 rows and checks. Historical statements above about
leaving original report tables unchanged describe the earlier investigation
stages, before this promotion.

## Superseded workspace cleanup

The project owner subsequently requested deletion of the superseded artifact
folders: `artifacts/h1/`, `artifacts/h6_and_h24/` and the initial
`artifacts/reproduction/`. Their historical paths above describe the evidence
reviewed before removal; the old fitted models are no longer available there.
Stable-a, stable-b, the original report snapshot and the small comparison
summaries remain retained. The two final comparison JSON summaries are tracked
in Git; other local diagnostic artifacts remain ignored.

## Current reviewer layout (12 November 2026)

The selected validated run was moved, without changing model/result values, to
`artifacts/final_run/`. Its exact datasets are now under `data/processed/`.
The duplicate stable-a/stable-b workspaces and large diagnostic trials were
removed after verification; the names above describe the historical comparison,
not current reviewer paths. Small final audit records are retained under
`reports/validation_checks/reproducibility/`. Original run metadata remains
immutable, and its historical input paths are translated by the explicit
relocation map in `reports/baseline.json`. See
`docs/reviewer_code_organisation.md` for the final folder table and report edits.
