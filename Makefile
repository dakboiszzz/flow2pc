PY ?= python

lint:
	$(PY) -m ruff check src/ tests/ main.py

test:
	$(PY) -m pytest -q

.PHONY: lint test
