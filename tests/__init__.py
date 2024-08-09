"""Copyright (c) Microsoft Corporation. Licensed under the MIT license."""

import os

if "HUGGINGFACE_REPO" not in os.environ:
    raise RuntimeError("The environment variable `HUGGINGFACE_REPO` must be set.")
