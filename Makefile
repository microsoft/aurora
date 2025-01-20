.PHONY: install test docs

install:
	pip install --upgrade pip
	pip install -e ".[dev]"
	pre-commit install

test:
	pytest tests -v --cov=aurora --cov-report=term --cov-report=html

docs:
	jupyter-book build docs
	cp -r docs/_extras/* docs/_build/html/
