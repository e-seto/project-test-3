PYTHON ?= python3

.PHONY: setup model

setup:
	$(PYTHON) -m pip install -r requirements.txt

model:
	$(PYTHON) src/05_modelling.py

