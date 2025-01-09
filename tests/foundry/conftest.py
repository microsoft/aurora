"""Copyright (c) Microsoft Corporation. Licensed under the MIT license."""

import json
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
def runner_process(azcopy_mock_work_dir):
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


@pytest.fixture(
    params=[
        "subprocess-local",
        "subprocess-blob",
        "docker-local",
        "docker-blob",
    ]
)
def mock_foundry_client(
    request,
    monkeypatch,
    requests_mock,
    tmp_path: Path,
) -> Generator[dict, None, None]:
    azcopy_mock_work_dir = tmp_path / "azcopy_work"

    if "subprocess" in request.param:
        # Already determine a possible working path for the mock of `azcopy`. It might not be used,
        # but we do already need to determine it.

        with runner_process(azcopy_mock_work_dir) as (
            p,
            stdin,
            stdout,
        ), mock_foundry_responses_subprocess(stdin, stdout, requests_mock):
            # Communicate via blob storage, so mock `azcopy` too.
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

            yield {
                "client_comm": BlobStorageCommunication(blob_url_with_sas),
                "host_comm": BlobStorageCommunication(blob_url_with_sas),
                "foundry_client": FoundryClient(MOCK_ADDRESS, "mock-token"),
            }

    elif "docker" in request.param:
        requests_mock.real_http = True
        client_comm_folder = tmp_path / "communication"

        # It's important to create the communication folder on the client side already. If we don't,
        # Docker will create it, and the permissions will then be wrong.
        client_comm_folder.mkdir(exist_ok=True, parents=True)
        azcopy_mock_work_dir.mkdir(exist_ok=True, parents=True)

        # Find the path of the server hook.
        server_hook = Path(__file__).parents[0] / "docker_server_hook.py"

        # Run the Docker container. Assume that it has already been built.
        azcopy_path = Path(__file__).parents[0] / "azcopy.py"
        p = subprocess.Popen(
            [
                "docker",
                "run",
                "-p",
                "5001:5001",
                "--rm",
                "-t",
                "-v",
                f"{client_comm_folder}:/communication",
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
                "aurora-foundry:latest",
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

            # Communicate via blob storage, so mock `azcopy` too.
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
