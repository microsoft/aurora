# Aurora on Azure AI Foundry

Aurora [is available as a model on Azure AI Foundry](https://ai.azure.com/explore/models)!
This part of the documentation describes how you can produce predictions with Aurora running on a Foundry endpoint.

## Managing Secrets

In order to access the endpoint on Azure AI Foundry,
you will need the endpoint URL and endpoint access token.
These can be found in the Azure interface.
[As will be explained later](/foundry/submission.md),
you will also need to create an URL to a Azure blob storage folder with a SAS token appended that has both read and write rights.
(In a nutshell, this blob storage folder is necessary to and retrieve data from the endpoint.)
Instead of storing these values in files, we recommend to store them in the environment variables `FOUNDRY_ENDPOINT`, `FOUNDRY_TOKEN`, and `BLOB_URL_WITH_SAS`.
This is what this documentation will assume.

### Accessing Environment Variables in a Jupyter Notebook

If your usual workflow is in a Jupyter notebook and you are having trouble setting and accessing environment variables, one alternative is to set the environment variables
in a file called `.env` and load them from there.
The layout of the `.env` file should follow this:

```
FOUNDRY_ENDPOINT=<foundry_endpoint>
FOUNDRY_TOKEN=<foundry_token>
BLOB_URL_WITH_SAS=<blob_url_with_sas>
```

Once this `.env` file has been created and populated, you can load it into a notebook environment with

```
!pip install python-dotenv
%load_ext dotenv
%dotenv path/to/.env
```

```{warning}
Do _not_ accidentally commit `.env` to version control.
```
