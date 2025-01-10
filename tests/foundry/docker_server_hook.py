"""Copyright (c) Microsoft Corporation. Licensed under the MIT license."""

import subprocess
import sys
from pathlib import Path
from urllib.parse import urlparse

# This will be run in the release Docker image, so packages required for mocking are not available.
subprocess.check_call([sys.executable, "-m", "pip", "install", "--no-deps", "requests_mock"])


import requests  # noqa: E402
import requests_mock  # noqa: E402

# First, mock requests that check for the existence of blobs.


def _matcher(request: requests.Request) -> requests.Response | None:
    """Mock requests that check for the existence of blobs."""
    url = urlparse(request.url)
    path = url.path[1:]  # Remove leading slash.

    if url.hostname and url.hostname.endswith("blob.core.windows.net"):
        # Assume that the local folder `/azcopy_work` is used by the mock of `azcopy`.
        local_path = Path("/azcopy_work") / path

        response = requests.Response()
        if local_path.exists():
            response.status_code = 200
        else:
            response.status_code = 404
        return response

    return None


mock = requests_mock.Mocker().__enter__()
mock.real_http = True
mock.add_matcher(_matcher)

from aurora.foundry.common.channel import BlobStorageCommunication  # noqa: E402

# Second, mock `azcopy`, assuming that the `azcopy` mock working directory is `/azcopy_work`.
BlobStorageCommunication._AZCOPY_EXECUTABLE = [
    "python",
    "/aurora_foundry/azcopy.py",
    "/azcopy_work",
]
