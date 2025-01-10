"""Copyright (c) Microsoft Corporation. Licensed under the MIT license."""

import json
import os
import re
import subprocess
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Generator

import pytest
import requests

from aurora.foundry.client.foundry import FoundryClient
from aurora.foundry.common.channel import BlobStorageCommunication

MOCK_ADDRESS = "https://mock-foundry.azurewebsites.net"


@contextmanager
def runner_process(azcopy_mock_work_dir: Path):
    score_script_path = Path(__file__).parents[2] / "aurora/foundry/server/score.py"
    runner_path = Path(__file__).parents[0] / "runner.py"
    p = subprocess.Popen(
        ["python", runner_path, azcopy_mock_work_dir, score_script_path],
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
def mock_foundry_responses_subprocess(stdin, stdout, requests_mock, base_address=MOCK_ADDRESS):
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


@pytest.fixture(params=["subprocess", "docker"])
def mock_foundry_client(
    request,
    monkeypatch,
    requests_mock,
    tmp_path: Path,
) -> Generator[dict, None, None]:
    # Communicate via blob storage, so mock `azcopy` too.
    azcopy_mock_work_dir = tmp_path / "azcopy_work"
    # It's important to already create the work folder. If we don't then the Docker image will
    # create it, and the permissions will then be wrong.
    azcopy_mock_work_dir.mkdir(exist_ok=True, parents=True)
    azcopy_path = Path(__file__).parents[0] / "azcopy.py"
    monkeypatch.setattr(
        BlobStorageCommunication,
        "_AZCOPY_EXECUTABLE",
        ["python", str(azcopy_path), str(azcopy_mock_work_dir)],
    )
    # The below test URL must start with `https`!
    blob_url_with_sas = "https://storageaccount.blob.core.windows.net/container/folder?SAS"

    def _matcher(request: requests.Request) -> requests.Response | None:
        """Mock requests that check for the existence of blobs."""
        if "blob.core.windows.net/" in request.url:
            # Split off the SAS token.
            path, _ = request.url.split("?", 1)
            # Split off the storage account URL.
            _, path = path.split("blob.core.windows.net/", 1)

            local_path = azcopy_mock_work_dir / path

            response = requests.Response()
            if local_path.exists():
                response.status_code = 200
            else:
                response.status_code = 404
            return response

        return None

    requests_mock.add_matcher(_matcher)

    if request.param == "subprocess":
        with runner_process(azcopy_mock_work_dir) as (p, stdin, stdout):  # noqa: SIM117
            with mock_foundry_responses_subprocess(stdin, stdout, requests_mock):
                yield {
                    "client_comm": BlobStorageCommunication(blob_url_with_sas),
                    "host_comm": BlobStorageCommunication(blob_url_with_sas),
                    "foundry_client": FoundryClient(MOCK_ADDRESS, "mock-token"),
                }

    elif request.param == "docker":
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
                f"{azcopy_mock_work_dir}:/azcopy_work",
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
                "client_comm": BlobStorageCommunication(blob_url_with_sas),
                "host_comm": BlobStorageCommunication(blob_url_with_sas),
                "foundry_client": FoundryClient("http://127.0.0.1:5001", "mock-token"),
            }

        finally:
            p.terminate()
            p.wait()

    else:
        raise ValueError(f"Bad Foundry mock mode: `{request.param}`.")
