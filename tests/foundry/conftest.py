"""Copyright (c) Microsoft Corporation. Licensed under the MIT license."""

import json
import traceback
from pathlib import Path
from queue import Queue
from threading import Thread
from typing import IO, Generator, Generic, TypeVar
from urllib.parse import urlparse

import pytest
import requests
from azure.storage.blob import BlobClient
from huggingface_hub import hf_hub_download

from aurora.foundry import BlobStorageChannel, FoundryClient
from aurora.foundry.server.mlflow_wrapper import AuroraModelWrapper

T = TypeVar("T")


class _Wrapped(Generic[T]):
    """Wrap an object to be accessible via `.item()`."""

    def __init__(self, x: T) -> None:
        self.x = x

    def item(self) -> T:
        return self.x


class _MockContext:
    """MLflow artifacts available in the tests."""

    artifacts = {
        "aurora-0.25-small-pretrained": hf_hub_download(
            repo_id="microsoft/aurora",
            filename="aurora-0.25-small-pretrained.ckpt",
        )
    }


def _server_work(queue_in: Queue[dict | None], queue_out: Queue[dict]) -> None:
    """Simulate a server. If it crashes, print the error.

    The server can be shut down by sending it `None` via `queue_in`.
    """
    try:
        context = _MockContext()

        model = AuroraModelWrapper()
        model.load_context(context)

        while message := queue_in.get():
            data = message["input_data"]["data"]
            response = model.predict(context, {"data": _Wrapped(data)})
            queue_out.put(response)
    except Exception:
        print("Server crashed with the following exception:")
        print(traceback.format_exc())


class _MockIO:
    """Simple mock of an `IO`."""

    def __init__(self, data: bytes) -> None:
        self.data = data

    def readall(self) -> bytes:
        return self.data


class _MockBlobClient:
    """Mock of `BlobClient` that reads from and writes to a local directory.

    Also checks for the presence of a SAS token.
    """

    def __init__(self, url: str, work_path: Path) -> None:
        url_parsed = urlparse(url)

        # Assert that a SAS token is present and right.
        assert url_parsed.query == "SAS"

        self.work_path = work_path / url_parsed.path.removeprefix("/")

    def upload_blob(self, f: IO, overwrite: bool) -> None:
        self.work_path.parent.mkdir(exist_ok=True, parents=True)
        with open(self.work_path, "wb") as local_f:
            local_f.write(f.read())

    def download_blob(self) -> _MockIO:
        with open(self.work_path, "rb") as local_f:
            return _MockIO(local_f.read())


@pytest.fixture()
def mock_foundry_client(
    request, monkeypatch, requests_mock, tmp_path
) -> Generator[dict, None, None]:
    blob_work_dir = tmp_path / "blob"

    # Communication queues for the sever thread:
    queue_in: Queue[dict | None] = Queue()
    queue_out: Queue[dict] = Queue()

    def _server_json(request, context) -> dict:
        message = json.loads(request.text)
        queue_in.put(message)
        response = queue_out.get()
        return response

    # Mock communication with the server:
    requests_mock.post("https://127.0.0.1", json=_server_json)

    def _mock_head(url: str) -> requests.Response:
        url_parsed = urlparse(url)
        path = url_parsed.path.removeprefix("/")

        response = requests.Response()
        response.status_code = 404

        if url_parsed.hostname and url_parsed.hostname.endswith(".blob.core.windows.net"):
            local_path = blob_work_dir / path

            if local_path.exists():
                response.status_code = 200

        return response

    # Mock HEAD requests that check for the existence of blobs:
    monkeypatch.setattr(requests, "head", _mock_head)

    def _mock_from_blob_url(url: str) -> object:
        return _MockBlobClient(url, blob_work_dir)

    # Mock `BlobClient`:
    monkeypatch.setattr(BlobClient, "from_blob_url", _mock_from_blob_url)

    # Start a separate thread that models the server.
    th = Thread(target=_server_work, args=(queue_in, queue_out))
    th.start()

    yield {
        "foundry_client": FoundryClient("https://127.0.0.1", "TOKEN"),
        "channel": BlobStorageChannel(
            "https://storageaccount.blob.core.windows.net/container/folder?SAS"
        ),
    }

    # Wait for the server to finish.
    queue_in.put(None)  # Kill signal
    th.join()
