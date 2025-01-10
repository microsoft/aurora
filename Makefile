.PHONY: install test docs docker-requirements docker swagger-file

DOCKER_WS ?= testwsacr
DOCKER_IMAGE ?= aurora-foundry:20250110-2

install:
	pip install --upgrade pip
	pip install -e ".[dev]"
	pre-commit install

test:
	DOCKER_IMAGE=$(DOCKER_WS).azurecr.io/$(DOCKER_IMAGE) pytest tests -v --cov=aurora --cov-report=term --cov-report=html

docs:
	jupyter-book build docs
	cp -r docs/_extras/* docs/_build/html/

docker-requirements: pyproject.toml
	(pip show pip-tools 1>/dev/null) || pip install pip-tools
	pip-compile --verbose --output-file _docker_requirements.txt pyproject.toml

docker:
	(pip show setuptools-scm 1>/dev/null) || pip install setuptools-scm
	AURORA_REPO_VERSION=`python -m setuptools_scm` docker build --build-arg AURORA_REPO_VERSION -t $(DOCKER_WS).azurecr.io/$(DOCKER_IMAGE) .

docker-acr:
	(pip show setuptools-scm 1>/dev/null) || pip install setuptools-scm
	[ ! -z "$(ACR)" ]
	AURORA_REPO_VERSION=`python -m setuptools_scm` az acr build --build-arg AURORA_REPO_VERSION -r "$(ACR)" -t $(DOCKER_IMAGE) .

swagger-file:
	pip install fastapi
	python aurora/foundry/server/generate_swagger.py aurora/foundry/server/swagger3.json
