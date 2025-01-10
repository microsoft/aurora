"""Copyright (c) Microsoft Corporation. Licensed under the MIT license.

A mock of the Azure ML inference server for more simple testing.
"""

import importlib.util as util
import json
import logging
import sys
from pathlib import Path
from urllib.parse import urlparse

import click
import requests
import requests_mock
from flask import Request
from werkzeug.test import EnvironBuilder

# Expose logging messages.
logger = logging.getLogger()
logger.setLevel("INFO")
stream_handler = logging.StreamHandler(sys.stderr)
formatter = logging.Formatter("[%(levelname)s] %(asctime)s %(name)s: %(message)s")
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)


@click.command()
@click.argument(
    "azcopy_mock_work_path",
    required=True,
    type=click.Path(
        exists=False, file_okay=False, dir_okay=True, resolve_path=True, path_type=Path
    ),
)
@click.argument(
    "path",
    required=True,
    type=click.Path(exists=True, file_okay=True, dir_okay=False, resolve_path=True, path_type=Path),
)
def main(azcopy_mock_work_path: Path, path: Path) -> None:
    spec = util.spec_from_file_location("score", path)
    assert spec is not None, "Could not load specification."
    score = util.module_from_spec(spec)
    assert score is not None, "Could not load module from specification."
    assert spec.loader is not None, "Specification has no loader."
    spec.loader.exec_module(score)

    # At this point, we mock `azcopy` too.
    azcopy_path = Path(__file__).parents[0] / "azcopy.py"
    sys.modules["aurora.foundry"].BlobStorageChannel._AZCOPY_EXECUTABLE = [
        "python",
        str(azcopy_path),
        str(azcopy_mock_work_path),
    ]

    def _matcher(request: requests.Request) -> requests.Response | None:
        """Mock requests that check for the existence of blobs."""
        url = urlparse(request.url)
        path = url.path[1:]  # Remove leading slash.

        if url.hostname and url.hostname.endswith(".blob.core.windows.net"):
            local_path = azcopy_mock_work_path / path

            local_path = azcopy_mock_work_path / path

            response = requests.Response()
            if local_path.exists():
                response.status_code = 200
            else:
                response.status_code = 404
            return response

        return None

    with requests_mock.Mocker() as mock:
        mock.real_http = True
        mock.add_matcher(_matcher)

        score.init()

        while True:
            method = sys.stdin.readline().strip()
            base_url = sys.stdin.readline().strip()
            query_params = sys.stdin.readline().strip()
            headers = json.loads(sys.stdin.readline().encode("utf-8").strip())
            payload = sys.stdin.readline().encode("utf-8").strip()

            builder = EnvironBuilder(
                method=method,
                base_url=base_url,
                query_string=query_params,
                headers=headers,
                data=payload,
            )
            env = builder.get_environ()
            flask_request = Request(env)

            resp = score.run(flask_request)
            answer = json.dumps(resp).encode("utf-8") if isinstance(resp, dict) else resp.data

            sys.stdout.write(answer.decode("utf-8"))
            sys.stdout.write("\n")
            sys.stdout.flush()


if __name__ == "__main__":
    main()
