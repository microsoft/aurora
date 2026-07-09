# Creating an Endpoint

```{warning}
Azure AI Foundry has deprecated model deployments based on MLflow. New deployments, including that of
Aurora 1.5, package the server-side code into a custom model and container image. Aurora remains
available on Foundry for all previous checkpoints, while the new model **Aurora-1.5** is
available with a custom model. This repository contains the necessary client-side code to call
and endpoint deployed with Foundry but does not currently support building your own endpoint, and
as such this page is deprecated.
```

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
