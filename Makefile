.PHONY: install test docs

install:
	pip install --upgrade pip
	pip install -e ".[dev]"

test:
	pytest tests -v --cov=aurora --cov-report=term --cov-report=html

docs:
	jupyter-book build
