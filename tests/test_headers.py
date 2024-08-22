"""Copyright (c) Microsoft Corporation. Licensed under the MIT license."""

from pathlib import Path

import pytest

COPYRIGHT_NOTICE: str = '"""Copyright (c) Microsoft Corporation. Licensed under the MIT license.'
"""str: Every file must start with this notice."""

PYTHON_FILES: list[Path] = []
"""list[Path]: Python files to scan for headers."""

_root = Path(__file__).parents[1]
for path in _root.rglob("**/*.py"):
    relative_path = path.relative_to(_root)

    # Ignore a possible virtual environment.
    if len(relative_path.parents) >= 2 and str(relative_path.parents[-2]) in {"venv"}:
        continue

    # Ignore the automatically generated version file.
    if relative_path.name in {"_version.py"}:
        continue

    PYTHON_FILES.append(path)


@pytest.mark.parametrize("python_file", PYTHON_FILES)
def test_presence_of_copyright_header(python_file: Path) -> None:
    with open(python_file) as f:
        lines = list(f.read().splitlines())

    if not lines or not lines[0].startswith(COPYRIGHT_NOTICE):
        raise AssertionError(f"`{python_file}` must start with the copyright notice.")
