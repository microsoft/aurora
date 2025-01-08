"""Copyright (c) Microsoft Corporation. Licensed under the MIT license."""

import abc
import json
import logging
import os
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Generator, Literal

from pydantic import BaseModel, HttpUrl

from aurora import Batch

__all__ = [
    "CommunicationChannel",
    "LocalCommunication",
    "BlobStorageCommunication",
    "iterate_prediction_files",
]

logger = logging.getLogger(__name__)


class CommunicationChannel:
    """A communication channel for sending very large files."""

    def send(self, batch: Batch, uuid: str, name: str) -> None:
        """Send `batch` under the file name `name` and mark the file as done.

        Args:
            batch (:class:`aurora.Batch`): Batch to send.
            uuid (str): UUID of the task.
            name (str): Name of `batch`.
        """
        name = f"{uuid}/{name}"
        self._send(batch, name)
        self._mark(name)

    def receive(self, uuid: str, name: str) -> Batch:
        """Receive the batch under the file name `name`.

        This function blocks until the file is ready.

        Args:
            uuid (str): UUID of the task.
            name (str): Name to receive.

        Returns:
            :class:`aurora.Batch`: Batch under the name `name`.
        """
        name = f"{uuid}/{name}"
        while not self._is_marked(name):
            time.sleep(0.5)
        return self._receive(name)

    @abc.abstractmethod
    def _send(self, batch: Batch, name: str) -> None:
        """Send `batch` under the file name `name` without marking the file.

        This method should be implemented.

        Args:
            batch (:class:`aurora.Batch`): Batch to send.
            name (str): Name of `batch`.
        """

    @abc.abstractmethod
    def _receive(self, name: str) -> Batch:
        """Receive the batch under the file name `name`.

        This function asserts that the file is ready and should be implemented by implementations.

        Args:
            name (str): Name to receive.

        Returns:
            :class:`aurora.Batch`: Batch under the file name `name`.
        """

    @abc.abstractmethod
    def _mark(self, name: str) -> None:
        """Mark the file `name` as done.

        Args:
            name (str): File to mark.
        """

    @abc.abstractmethod
    def _is_marked(self, name: str) -> bool:
        """Check whether the file `name` has been marked.

        Args:
            name (str): File to check.

        Returns:
            bool: Whether `name` has been marked or not.
        """

    @abc.abstractmethod
    def to_spec(self) -> dict[str, str]:
        """Convert this channel to specification that can be serialised into JSON.

        Returns:
            dict[str, str]: Specification.
        """


class LocalCommunication(CommunicationChannel):
    """A communication channel via a local folder."""

    def __init__(self, folder: str | Path) -> None:
        """Instantiate.

        Args:
            folder (str or Path): Folder to use.
        """
        self.folder = Path(folder)

    def to_spec(self) -> dict[str, str]:
        return {
            "class_name": "LocalCommunication",
            "folder": str(self.folder),
        }

    class Spec(BaseModel):
        class_name: Literal["LocalCommunication"]
        folder: Path

        def construct(self) -> "LocalCommunication":
            return LocalCommunication(folder=str(self.folder))

    def _send(self, batch: Batch, name: str) -> None:
        target = self.folder / name
        target.parent.mkdir(exist_ok=True, parents=True)
        batch.to_netcdf(target)

    def _receive(self, name: str) -> Batch:
        return Batch.from_netcdf(self.folder / name)

    def _mark(self, name: str) -> None:
        target = self.folder / f"{name}.finished"
        target.parent.mkdir(exist_ok=True, parents=True)
        target.touch()

    def _is_marked(self, name: str) -> bool:
        return (self.folder / f"{name}.finished").exists()


class BlobStorageCommunication(CommunicationChannel):
    """A communication channel via a folder in a Azure Blob Storage container."""

    _AZCOPY_EXECUTABLE: list[str] = ["azcopy"]

    def __init__(self, blob_folder: str) -> None:
        """Instantiate.

        Args:
            blob_folder (str): Folder to use. This must be a full URL that includes a SAS token
                with read and write permissions.
        """
        self.blob_folder = blob_folder
        if "?" not in blob_folder:
            raise ValueError("Given URL does not appear to contain a SAS token.")

    def to_spec(self) -> dict[str, str]:
        return {
            "class_name": "BlobStorageCommunication",
            "blob_folder": self.blob_folder,
        }

    class Spec(BaseModel):
        class_name: Literal["BlobStorageCommunication"]
        blob_folder: HttpUrl  # TODO: Can we force this to be `https`?

        def construct(self) -> "BlobStorageCommunication":
            return BlobStorageCommunication(blob_folder=str(self.blob_folder))

    def _blob_path(self, name: str) -> str:
        """For a given file name `name`, get the full path including the SAS token.

        Args:
            name (str): File name.

        Returns:
            str: Full path including the SAS token.
        """
        url, sas = self.blob_folder.split("?", 1)
        # Don't use `pathlib.Path` for web URLs! It messes up the protocol.
        return f"{os.path.join(url, name)}?{sas}"

    def _azcopy(self, command: list[str]) -> str:
        result = subprocess.run(
            self._AZCOPY_EXECUTABLE + command,
            stdout=subprocess.PIPE,
            check=True,
        )
        return result.stdout.decode("utf-8")

    def _send(self, batch: Batch, name: str) -> None:
        with tempfile.NamedTemporaryFile() as tf:
            batch.to_netcdf(tf.name)
            self._azcopy(["copy", tf.name, self._blob_path(name)])

    def _receive(self, name: str) -> Batch:
        with tempfile.NamedTemporaryFile() as tf:
            self._azcopy(["copy", self._blob_path(name), tf.name])
            return Batch.from_netcdf(tf.name)

    def _mark(self, name: str) -> None:
        with tempfile.TemporaryDirectory() as td:
            mark_file_path = Path(td) / f"{name}.finished"
            mark_file_path.parent.mkdir(exist_ok=True, parents=True)
            mark_file_path.touch()
            self._azcopy(["copy", str(mark_file_path), self._blob_path(f"{name}.finished")])

    def _is_marked(self, name: str) -> bool:
        out = json.loads(
            self._azcopy(
                [
                    "list",
                    self._blob_path(f"{name}.finished"),
                    "--output-type",
                    "json",
                    "--output-level",
                    "essential",
                ]
            )
        )
        return (
            len(out) == 2
            and isinstance(out[0], dict)
            and out[0].get("MessageType", None) == "ListObject"
            and isinstance(out[1], dict)
            and out[1].get("MessageType", None) == "EndOfJob"
        )


def iterate_prediction_files(name: str, num_steps: int) -> Generator[str, None, None]:
    """For a file name `name`, generate `num_steps` variations of `name`: one for every prediction
    step.

    Args:
        name (str): Base file name to derive the file names of the predictions from.
        num_steps (int): Number of prediction steps.

    Yields:
        str: For every prediction, the derived file name.
    """
    name, ext = os.path.splitext(name)
    for i in range(num_steps):
        yield f"{name}-{i:03d}{ext}"
