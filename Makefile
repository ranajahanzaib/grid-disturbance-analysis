.PHONY: install test lint format clean check run-app prepare-data

PYTHON ?= .venv/bin/python
PIP ?= .venv/bin/pip
PYTEST ?= .venv/bin/pytest
RUFF ?= .venv/bin/ruff
MYPY ?= .venv/bin/mypy

install:
	uv venv --python 3.11 .venv
	uv pip install --python $(PYTHON) -e ".[dev]"

test:
	MPLCONFIGDIR=/tmp/matplotlib $(PYTEST) --cov=grid_disturbance --cov-report=term-missing

lint:
	$(RUFF) check src tests scripts examples
	$(MYPY) src/grid_disturbance

format:
	$(RUFF) format src tests scripts examples
	$(RUFF) check --fix src tests scripts examples

prepare-data:
	$(PYTHON) scripts/prepare_dataset.py --samples 50

run-app:
	$(PYTHON) -m streamlit run src/grid_disturbance/app/main.py

clean:
	rm -rf .pytest_cache .ruff_cache .mypy_cache .coverage htmlcov build dist *.egg-info
