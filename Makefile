PYTHON ?= python
.PHONY: test validate-data verify reports reproduce

test:
	$(PYTHON) -W ignore::DeprecationWarning -m unittest discover -s tests -v
validate-data:
	$(PYTHON) -m data_processing.build_dataset
verify:
	$(PYTHON) -W ignore::DeprecationWarning -m modelling.validation.verify_model_results
reports:
	$(PYTHON) -m reports.reproduce

# Rebuild from retained sources, train all models, and generate fresh reports.
reproduce:
	$(PYTHON) -m reports.reproduce --full
