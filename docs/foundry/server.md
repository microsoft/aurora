# Running the Inference Server

Build the Docker image:

```bash
make docker
```

Upload the resulting image to Azure AI foundry.

Building the Docker image depends on a list of precompiled dependencies.
If you change the requirements in `pyproject.toml`, this list must be updated:

```bash
make docker-requirements
```
