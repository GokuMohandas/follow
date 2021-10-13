# Makefile
SHELL := /bin/bash

.PHONY: help
help:
	@echo "Commands:"
	@echo "venv    : creates development environment."
	@echo "style   : runs style formatting."
	@echo "clean   : cleans all unnecessary files."
	@echo "test    : run data and code tests."
	@echo "dvc     : add and push versioned data pointer files."

# Environment
.ONESHELL:
venv:
	python3 -m venv venv
	source venv/bin/activate && \
	python -m pip install --upgrade pip setuptools wheel && \
	python -m pip install -e ".[dev]" && \
	pre-commit install && \
	pre-commit autoupdate && \
	echo "Be sure to execute 'source venv/bin/activate' to use the virtual environment"

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

# DVC
.PHONY: dvc
dvc:
	dvc add data/projects.json
	dvc add data/tags.json
	dvc add data/features.json
	dvc add data/features.parquet
	dvc push
