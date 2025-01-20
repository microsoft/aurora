# Creating an Endpoint

Likely you don't need to create an endpoint yourself,
because Aurora is already available in the [Azure AI model catalog](https://ai.azure.com/explore/models).

Nevertheless, should you want to create an endpoint,
then you can follow these instructions.
The model is served via [MLflow](https://mlflow.org/).
First, make sure that `mlflow` is installed:

```bash
pip install mlflow
```

Then build the MLflow model as follows:

```bash
python package_mlflow.py
```
