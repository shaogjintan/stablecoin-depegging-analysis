# Superseded data-processing diagnostics

Original files are preserved in
`delete/archive_cleanup/history/obsolete_data_processing_checks/`:

- coverage_table.py: earlier separate market/on-chain coverage reporting using
  historical common_sample flags. The maintained exporter is
  modelling/reporting/coverage_table.py, with explicit full-span denominators.
- verify_embargo_and_episodes.py: earlier autocorrelation and episode summaries
  plus split checks. Its claim that ACF proves a required embargo duration is
  not a conclusion endorsed by retaining the script. Current boundary checks
  enforce the configured purge/embargo and episode exclusions; they do not
  establish independence or statistically prove that 24h is sufficient.

Neither archived script has maintained callers. Current verification code and
its tests are the supported checks for the final saved runs.
