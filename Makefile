# Makefile
SHELL := /bin/bash

.PHONY: help
help:
	@echo "Commands:"
	@echo "venv    : creates development environment."
	@echo "style   : runs style formatting."
	@echo "clean   : cleans all unnecessary files."

# Environment
.ONESHELL:
venv:
	python3 -m venv venv
	source venv/bin/activate && \
	python -m pip install --upgrade pip setuptools wheel && \
	python -m pip install -e ".[dev]" --no-cache-dir && \
	pre-commit install && \
	pre-commit autoupdate

# Styling
.PHONY: style
style:
	black .
	flake8
	isort .

# Cleaning
.PHONY: clean
clean: style
	find . -type f -name "*.DS_Store" -ls -delete
	find . | grep -E "(__pycache__|\.pyc|\.pyo)" | xargs rm -rf
	find . | grep -E ".pytest_cache" | xargs rm -rf
	find . | grep -E ".ipynb_checkpoints" | xargs rm -rf
	rm -f .coverage

# Test
.PHONY: test
test:
	cd tests && great_expectations checkpoint run projects
	cd tests && great_expectations checkpoint run tags
	pytest -m "not training"