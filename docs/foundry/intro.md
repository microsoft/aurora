# Aurora on Azure AI Foundry

Aurora [is available as a model on Azure AI Foundry](https://ai.azure.com/explore/models)!
This part of the documentation describes how you can produce predictions with Aurora running on a Foundry endpoint.

## Accessing Azure AI Foundry
In order to access the endpoint on Azure AI Foundry, it is preferable to set environment variables, for example in a `.env`. This includes Azure AI Foundry authentication, and blob storage authentication. The intermediary blob storage is necessary due to the file size of initial conditions and predictions. The layout of the `.env` file should follow:

```
FOUNDRY_ENDPOINT=<foundry_endpoint>
FOUNDRY_TOKEN=<foundry_token>
BLOB_URL_WITH_SAS=<blob_url_with_sas>
```

### Running on Azure AI Foundry in a Jupyter Notebook

In a Jupyter notebook, make sure the environment variables are set. If you have created a `.env` file, you can load these to the notebook environment with:

```
!pip install python-dotenv
%load_ext dotenv
%dotenv ./relative/path/to/file/.env
```
