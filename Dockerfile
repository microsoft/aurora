# Use an official Python runtime as a parent image.
FROM mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04:latest

WORKDIR /aurora_foundry
COPY ./pyproject.toml .

# Assuming dependencies are fairly fixed, we can install them first and then copy the rest of the
# code to avoid re-installing dependencies when the code changes.
COPY _docker_requirements.txt .
RUN pip install --upgrade pip virtualenv && \
    virtualenv venv -p python3.10 && \
    . venv/bin/activate && \
    pip install -r _docker_requirements.txt

# Download model weights.
RUN ./venv/bin/python -c 'from huggingface_hub import hf_hub_download; hf_hub_download(repo_id="microsoft/aurora", filename="aurora-0.25-small-pretrained.ckpt")' && \
    ./venv/bin/python -c 'from huggingface_hub import hf_hub_download; hf_hub_download(repo_id="microsoft/aurora", filename="aurora-0.25-finetuned.ckpt")'

COPY ./aurora ./aurora
COPY ./LICENSE.txt .
COPY ./README.md .

ARG AURORA_REPO_VERSION
RUN [ ! -z "${AURORA_REPO_VERSION}" ] || { echo "AURORA_REPO_VERSION must be set."; exit 1; } && \
    . venv/bin/activate && \
    SETUPTOOLS_SCM_PRETEND_VERSION="$AURORA_REPO_VERSION" pip install -e .

# Install `azcopy` and the AML inference server.
RUN wget https://aka.ms/downloadazcopy-v10-linux -O azcopy.tar.gz && \
    cp $(tar -xvzf azcopy.tar.gz | grep azcopy$) /usr/local/bin/azcopy
RUN . ./venv/bin/activate && \
    pip install azureml-inference-server-http

# Make port 5001 available to the world outside this container.
EXPOSE 5001
ENV PORT=5001

CMD ["./venv/bin/azmlinfsrv", "--entry_script", "aurora/foundry/server/score.py"]
