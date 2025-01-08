.PHONY: install test docs docker-requirements docker

install:
	pip install --upgrade pip
	pip install -e ".[dev]"
	pre-commit install

test:
	pytest tests -v --cov=aurora --cov-report=term --cov-report=html

docs:
	jupyter-book build docs
	cp -r docs/_extras/* docs/_build/html/

docker-requirements: pyproject.toml
	(pip show pip-tools 1>/dev/null) || pip install pip-tools
	pip-compile --verbose --output-file _docker_requirements.txt pyproject.toml

docker:
	(pip show setuptools-scm 1>/dev/null) || pip install setuptools-scm
	AURORA_REPO_VERSION=`python -m setuptools_scm` docker build --build-arg AURORA_REPO_VERSION -t aurora-foundry:latest .
