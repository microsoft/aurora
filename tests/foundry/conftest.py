"""Copyright (c) Microsoft Corporation. Licensed under the MIT license."""

import json
import os
import re
import subprocess
import time
from contextlib import contextmanager
from pathlib import Path
from typing import IO, Generator, Tuple
from urllib.parse import urlparse

import pytest
import requests

from aurora.foundry.client.foundry import FoundryClient
from aurora.foundry.common.channel import BlobStorageChannel

MOCK_ADDRESS = "https://mock-foundry.azurewebsites.net"


@contextmanager
def runner_process(
    azcopy_mock_work_path: Path | None,
) -> Generator[Tuple[subprocess.Popen, IO, IO], None, None]:
    """Launch a runner process that mocks the Azure ML Inference Server."""
    score_script_path = Path(__file__).parents[2] / "aurora/foundry/server/score.py"
    runner_path = Path(__file__).parents[0] / "runner.py"
    p = subprocess.Popen(
        [
            "python",
            runner_path,
            *(
                ["--azcopy-mock-work-path", str(azcopy_mock_work_path)]
                if azcopy_mock_work_path
                else []
            ),
            score_script_path,
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
    )
    stdin = p.stdin
    stdout = p.stdout
    assert stdin is not None and stdout is not None
    yield p, stdin, stdout
    p.terminate()
    p.wait()


@contextmanager
def mock_foundry_responses_subprocess(
    stdin: IO, stdout: IO, requests_mock, base_address: str = MOCK_ADDRESS
) -> Generator[None, None, None]:
    """Mock requests to Foundry by redirecting them to the subprocess."""

    def _mock_send(request, context) -> dict:
        method = request.method.encode("unicode_escape")
        text = request.text or ""
        stdin.write(method + b"\n")
        stdin.write(request.path.encode("unicode_escape") + b"\n")
        stdin.write(request.url.partition("?")[2].encode("unicode_escape") + b"\n")
        stdin.write(json.dumps(dict(request.headers)).encode("unicode_escape") + b"\n")
        stdin.write(text.encode("unicode_escape") + b"\n")
        stdin.flush()

        output = stdout.readline()
        if not output:
            raise RuntimeError("Runner returned no answer. It likely crashed.")

        return json.loads(output.decode("unicode_escape"))

    requests_mock.post(
        f"{base_address}/score",
        json=_mock_send,
    )
    requests_mock.get(
        re.compile(rf"{base_address}/score\?task_id=.*"),
        json=_mock_send,
    )
    yield


def mock_azcopy(tmp_path: Path, monkeypatch, requests_mock) -> Tuple[Path, Path, str]:
    """Mock `azcopy`."""
    # Communicate via blob storage, so mock `azcopy` too.
    azcopy_mock_work_path = tmp_path / "azcopy_work"
    # It's important to already create the work folder. If we don't then the Docker image will
    # create it, and the permissions will then be wrong.
    azcopy_mock_work_path.mkdir(exist_ok=True, parents=True)
    azcopy_path = Path(__file__).parents[0] / "azcopy.py"
    monkeypatch.setattr(
        BlobStorageChannel,
        "_AZCOPY_EXECUTABLE",
        ["python", str(azcopy_path), str(azcopy_mock_work_path)],
    )
    # The below test URL must start with `https`!
    blob_url_with_sas = "https://storageaccount.blob.core.windows.net/container/folder?SAS"

    def _matcher(request: requests.Request) -> requests.Response | None:
        """Mock requests that check for the existence of blobs."""
        url = urlparse(request.url)
        path = url.path[1:]  # Remove leading slash.

        if url.hostname and url.hostname.endswith(".blob.core.windows.net"):
            local_path = azcopy_mock_work_path / path

            response = requests.Response()
            if local_path.exists():
                response.status_code = 200
            else:
                response.status_code = 404
            return response

        return None

    requests_mock.add_matcher(_matcher)

    return azcopy_path, azcopy_mock_work_path, blob_url_with_sas


@pytest.fixture(
    params=[
        "subprocess",
        "subprocess-real-container",
        "docker",
    ]
)
def mock_foundry_client(
    request,
    tmp_path: Path,
    monkeypatch,
    requests_mock,
) -> Generator[dict, None, None]:
    if request.param == "subprocess":
        azcopy_path, azcopy_mock_work_path, blob_url_with_sas = mock_azcopy(
            tmp_path, monkeypatch, requests_mock
        )

        with runner_process(azcopy_mock_work_path) as (p, stdin, stdout):  # noqa: SIM117
            with mock_foundry_responses_subprocess(stdin, stdout, requests_mock):
                yield {
                    "channel": BlobStorageChannel(blob_url_with_sas),
                    "foundry_client": FoundryClient(MOCK_ADDRESS, "mock-token"),
                }

    elif request.param == "subprocess-real-container":
        requests_mock.real_http = True

        if "TEST_BLOB_URL_WITH_SAS" not in os.environ:
            pytest.skip("`TEST_BLOB_URL_WITH_SAS` is not set, so test cannot be run.")
        blob_url_with_sas = os.environ["TEST_BLOB_URL_WITH_SAS"]

        with runner_process(None) as (p, stdin, stdout):  # noqa: SIM117
            with mock_foundry_responses_subprocess(stdin, stdout, requests_mock):
                yield {
                    "channel": BlobStorageChannel(blob_url_with_sas),
                    "foundry_client": FoundryClient(MOCK_ADDRESS, "mock-token"),
                }

    elif request.param == "docker":
        azcopy_path, azcopy_mock_work_path, blob_url_with_sas = mock_azcopy(
            tmp_path, monkeypatch, requests_mock
        )

        requests_mock.real_http = True

        if "DOCKER_IMAGE" not in os.environ:
            raise RuntimeError(
                "Set the environment variable `DOCKER_IMAGE` "
                "to the release image of Aurora Foundry."
            )
        docker_image = os.environ["DOCKER_IMAGE"]

        # Run the Docker container. Assume that it has already been built. Insert the hook
        # to mock things on the server side.
        server_hook = Path(__file__).parents[0] / "docker_server_hook.py"
        p = subprocess.Popen(
            [
                "docker",
                "run",
                "-p",
                "5001:5001",
                "--rm",
                "-t",
                "-v",
                f"{azcopy_mock_work_path}:/azcopy_work",
                "--mount",
                f"type=bind,src={azcopy_path},dst=/aurora_foundry/azcopy.py,readonly",
                "--mount",
                (
                    f"type=bind"
                    f",src={server_hook}"
                    f",dst=/aurora_foundry/aurora/foundry/server/_hook.py"
                    f",readonly"
                ),
                docker_image,
            ],
        )
        try:
            # Wait for the server to come online.
            start = time.time()
            while True:
                try:
                    res = requests.get("http://127.0.0.1:5001/")
                    res.raise_for_status()
                except (requests.ConnectionError, requests.HTTPError) as e:
                    # Try for at most 10 seconds.
                    if time.time() - start < 10:
                        time.sleep(0.5)
                        continue
                    else:
                        raise e
                break

            yield {
                "channel": BlobStorageChannel(blob_url_with_sas),
                "foundry_client": FoundryClient("http://127.0.0.1:5001", "mock-token"),
            }

        finally:
            p.terminate()
            p.wait()

    else:
        raise ValueError(f"Bad Foundry mock mode: `{request.param}`.")
