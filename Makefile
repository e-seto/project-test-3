PYTHON ?= python3

.PHONY: setup model

setup:
	$(PYTHON) -m pip install -r requirements.txt

model:
	$(PYTHON) -m src.05_main

