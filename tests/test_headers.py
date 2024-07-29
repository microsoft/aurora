# Copyright (c) Microsoft Corporation. Licensed under the MIT license.

from pathlib import Path

import pytest

SKIP_FILES: set[str] = {"_version.py"}
"""set[str]: These files are not required to have a copyright notice."""

COPYRIGHT_NOTICE = "# Copyright (c) Microsoft Corporation. Licensed under the MIT license."


@pytest.mark.parametrize("python_file", Path(__file__).parents[1].rglob("**/*.py"))
def test_presence_of_copyright_header(python_file: Path) -> None:
    if python_file.name in SKIP_FILES:
        return

    with open(python_file) as f:
        lines = list(f.read().splitlines())

    if not lines or lines[0].strip() != COPYRIGHT_NOTICE:
        raise AssertionError(f"The first line of `{python_file}` must be the copyright notice.")
