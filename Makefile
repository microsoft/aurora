.PHONY: install test docs docker-requirements docker swagger-file

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
	AURORA_REPO_VERSION=`python -m setuptools_scm` docker build --build-arg AURORA_REPO_VERSION -t testwsacr.azurecr.io/aurora-foundry:20250110-1 .

docker-acr:
	(pip show setuptools-scm 1>/dev/null) || pip install setuptools-scm
	[ ! -z "$(ACR)" ]
	AURORA_REPO_VERSION=`python -m setuptools_scm` az acr build --build-arg AURORA_REPO_VERSION -r "$(ACR)" -t aurora-foundry:20250110-1 .

swagger-file:
	pip install fastapi
	python aurora/foundry/server/generate-swagger.py aurora/foundry/server/swagger3.json
