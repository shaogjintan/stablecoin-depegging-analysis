# Historical training and threshold analyses

Three obsolete scripts are preserved under delete/archive_cleanup/history/:

- modelling/train_logreg_yearcv.py: an earlier combined LR training, calendar
  CV, scoring and reporting runner. Its _calendar_cv_folds implementation was
  verified AST-identical to modelling/training/calendar_cv.py before moving.
  Maintained C selection is in logistic_tuning.py; protocol orchestration is
  in train_models.py. The old runner is not the current reproduction route.
- analysis/local_curve.py: sweep validation false-alarm budgets across earlier
  model configurations. This is distinct from changing outcome definitions.
- analysis/ex_post_comparison.py: compare validation-selected cutoffs against
  cutoffs selected using test-year outcomes. The latter is retrospective and
  not a deployable out-of-sample threshold policy. This was exploratory
  supervisor-requested context, not the final live prediction protocol.

Neither analysis script has maintained callers; both reference retired modules.
This note preserves their purpose without making historical numbers or methods
part of current results. The original files remain intact in quarantine.

## Other superseded training runners

The following originals are also preserved under
`delete/archive_cleanup/history/modelling/`:

- corrected_h1.py: separate supplementary h1 LR/RF/baseline runner, importing
  the earlier combined LR/RF workflow.
- corrected_lr_rf.py: earlier combined LR/RF fitting, CV, calibration, scoring,
  partition preparation and aggregation. Its default threshold policy was
  max_validation_recall, unlike the final exhaustive_lowest policy.
- refit_final.py: refit stage consuming separate earlier model and cutoff runs.
  It copied saved cutoffs rather than selecting them within one unified run.

The maintained train_models.py handles all three models and horizons with the
practice/calibration/refit/test protocol. These historical runners are not
claimed to be exact full-script duplicates. Their orchestration and defaults
are superseded. Original source snapshots inside final artifact folders remain
unchanged and are the code provenance for those specific saved runs.

## No-refit source copies

Original historical experiment sources are retained with their artifacts:

- artifacts/sensitivity/random_forest_xgboost_no_refit/original_runner/runner_source.py
- artifacts/sensitivity/xgboost_no_refit/original_source/source/corrected_xgb.py

These paths are relative to archive/. Later refactored copies formerly
under modelling/sensitivity/ were not byte-identical: they changed imports,
extracted shared helpers, and added threshold-policy options. Those unused
copies are preserved in delete/archive_cleanup/history/no_refit_refactored_copies/.
The retained original snapshots remain unchanged. No equivalence of complete
runners or historical results is asserted by removing the redundant location.
