"""Copyright (c) Microsoft Corporation. Licensed under the MIT license.

A mock of `azcopy` designed specifically for the tests here.
"""

import json
import logging
import shutil
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


def _parse_path(path: str, work_path: Path) -> Path:
    if path.startswith("https://"):
        path, _ = path.split("?", 1)  # Split off the SAS token.
        _, path = path.split("blob.core.windows.net/", 1)  # Split off the storage account URL.
        return work_path / path
    else:
        # Just a local path.
        return Path(path)


@click.command(context_settings=dict(ignore_unknown_options=True))
@click.argument(
    "work_path",
    required=True,
    type=click.Path(
        exists=False, file_okay=False, dir_okay=True, resolve_path=True, path_type=Path
    ),
)
@click.argument("args", nargs=-1, type=click.UNPROCESSED)
def main(work_path: Path, args: tuple[str, ...]) -> None:
    assert len(args) >= 1

    logger.info(f'Faking `azcopy` call: `azcopy {" ".join(args)}`.')

    if args[0] in {"ls", "list"}:
        assert len(args) >= 2

        path = _parse_path(args[1], work_path)

        out: list[dict[str, str]] = []
        if path.exists():
            out.append({"MessageType": "ListObject"})
        for _ in path.rglob("*"):
            out.append({"MessageType": "ListObject"})
        out.append({"MessageType": "EndOfJob"})

        print(json.dumps(out))

    elif args[0] in {"cp", "copy"}:
        assert len(args) == 3

        source = _parse_path(args[1], work_path)
        target = _parse_path(args[2], work_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(source, target)

    else:
        raise RuntimeError(f"Unknown command `{args[0]}`.")


if __name__ == "__main__":
    main()
