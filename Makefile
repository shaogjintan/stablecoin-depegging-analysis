PYTHON ?= python
.PHONY: test validate-data verify reports

test:
	$(PYTHON) -W ignore::DeprecationWarning -m unittest discover -s tests -v
validate-data:
	$(PYTHON) -m data_processing.build_dataset
verify:
	$(PYTHON) -W ignore::DeprecationWarning -m modelling.verify_release
reports:
	$(PYTHON) -m scripts.reproduce
