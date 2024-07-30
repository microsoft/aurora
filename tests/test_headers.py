"""Copyright (c) Microsoft Corporation. Licensed under the MIT license."""

from pathlib import Path

import pytest

SKIP_FILES: set[str] = {"_version.py"}
"""set[str]: These files are not required to have a copyright notice."""

COPYRIGHT_NOTICE: str = '"""Copyright (c) Microsoft Corporation. Licensed under the MIT license.'


@pytest.mark.parametrize("python_file", Path(__file__).parents[1].rglob("**/*.py"))
def test_presence_of_copyright_header(python_file: Path) -> None:
    if python_file.name in SKIP_FILES:
        return

    with open(python_file) as f:
        lines = list(f.read().splitlines())

    if not lines or not lines[0].startswith(COPYRIGHT_NOTICE):
        raise AssertionError(f"`{python_file}` must start with the copyright notice.")
