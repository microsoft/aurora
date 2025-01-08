"""Copyright (c) Microsoft Corporation. Licensed under the MIT license.

A mock of the Azure ML inference server for more simple testing.
"""

import importlib.util as util
import json
import logging
import sys
from pathlib import Path

import click

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
    sys.modules["aurora.foundry"].BlobStorageCommunication._AZCOPY_EXECUTABLE = [
        "python",
        str(azcopy_path),
        str(azcopy_mock_work_path),
    ]

    score.init()

    while True:
        raw_data = sys.stdin.readline()
        raw_data = raw_data.encode("utf-8").decode("unicode_escape")

        answer = json.dumps(score.run(raw_data))

        sys.stdout.write(answer.encode("unicode_escape").decode("utf-8"))
        sys.stdout.write("\n")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
