FROM nvcr.io/nvidia/pytorch:25.08-py3
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app
SHELL ["/bin/bash", "-c"]

# Copy over source, create the environment, and install the repo in editable
# mode.
RUN mkdir -p /app/aurora/aurora
COPY pyproject.toml LICENSE.txt /app/aurora
RUN touch /app/aurora/__init__.py \
    && touch /app/aurora/README.md \
    && uv venv --python 3.13 \
    && SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0 uv pip install -e /app/aurora

# Use the environment automatically.
ENV VIRTUAL_ENV="/app/.venv/"
ENV PATH="/app/.venv/bin:$PATH"

# Let the user enter at `/app/aurora`.
WORKDIR /app/aurora
