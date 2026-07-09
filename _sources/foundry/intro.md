# Aurora on Azure AI Foundry

Aurora [is available as a model on Azure AI Foundry](https://ai.azure.com/explore/models)!
This part of the documentation describes how you can produce predictions with Aurora running on a Foundry endpoint.

## Available versions

Two versions of Aurora models are available on Foundry. The original model "Aurora" contains the original
published checkpoints and retains compatibility with existing Aurora workflows, while "Aurora-1.5"
includes only the new Aurora 1.5 checkpoints. Additionally, Aurora 1.5 supports a few other
server-side features intended to improve quality of life when using the Foundry model. These arguments
may be passed to the client API interface `aurora.foundry.submit`:
- `fine_lead_times`: Sub-step lead times in hours within each main step. This feature enables the
higher temporal resolution up to 1 hour.
- `saved_surf_vars`, `saved_atmos_vars`, `saved_atmos_levels`, and `saved_static_vars`: These options
reduce the number of output variables and/or levels saved in the final predictions. If only some variables are
desired rather than the complete set, this can linearly reduce the total data volume and speed up
server-side data generation and client-side data downloading.
- `async_upload_workers`: Setting this to a value > 0 is strongly recommended as it will enable
the server to write the data output asynchronously alongside the compute inference loop. Adding more
workers can allow the data loop to keep up with inference, but too many may result in out-of-memory,
hangs, or other stability issues, depending on task configuration. Values up to 8 should work on an
A100 GPU instance and may reduce job time by 5x or more.

## Managing Secrets

In order to access the endpoint on Azure AI Foundry,
you will need the endpoint URL and endpoint access token.
These can be found in the Azure interface.
[As will be explained later](submission.md),
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
