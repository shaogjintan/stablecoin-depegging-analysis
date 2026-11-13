# Automated checks

Run `make test` or `python -m unittest discover -s tests -v` from the repository
root. Tests cover labels, feature calculations, chronological splits, matched
samples, training, calibration, alarms, report exporters and workflow isolation.
Tests use synthetic or temporary inputs rather than retraining the study models.

`diagnose_onchain_features.py` is a manual diagnostic, not a unit test.
Saved-model evidence is checked separately with
`python -m modelling.validation.verify_model_results --check-model-scores`.
