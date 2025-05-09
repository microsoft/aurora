"""Copyright (c) Microsoft Corporation. Licensed under the MIT license."""

import abc
import logging
import os
import tempfile
import time
from pathlib import Path
from typing import Generator, Literal

import requests
from azure.storage.blob import BlobClient
from pydantic import BaseModel, HttpUrl

from aurora import Batch

__all__ = [
    "CommunicationChannel",
    "BlobStorageChannel",
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

    def receive(self, uuid: str, name: str, timeout: int = 120) -> Batch:
        """Receive the batch under the file name `name`.

        This function blocks until the file is ready.

        Args:
            uuid (str): UUID of the task.
            name (str): Name to receive.
            timeout (int, optional): Timeout in seconds. Defaults to 2 minutes.

        Returns:
            :class:`aurora.Batch`: Batch under the name `name`.
        """
        name = f"{uuid}/{name}"
        start = time.time()
        while not self._is_marked(name):
            if time.time() - start < timeout:
                time.sleep(1)
            else:
                raise TimeoutError("File was not marked within the timeout.")
        return self._receive(name)

    def write(self, data: bytes, uuid: str, name: str) -> None:
        """Write `data` to `name`.

        Args:
            Data (bytes): Data to write.
            uuid (str): UUID of the task.
            name (str): Name to write to.
        """
        name = f"{uuid}/{name}"
        self._write(data, name)
        self._mark(name)

    def read(self, uuid: str, name: str, timeout: int = 120) -> bytes:
        """Read `name`.

        Args:
            uuid (str): UUID of the task.
            name (str): Name to read.
            timeout (int, optional): Timeout in seconds. Defaults to 2 minutes.

        Returns:
            bytes: Data of `name`.
        """
        name = f"{uuid}/{name}"
        start = time.time()
        while not self._is_marked(name):
            if time.time() - start < timeout:
                time.sleep(1)
            else:
                raise TimeoutError("File was not marked within the timeout.")
        return self._read(name)

    def exists(self, uuid: str, name: str) -> bool:
        """Check whether `name` is available.

        Args:
            uuid (str): UUID of the task.
            name (str): Name to check for.

        Returns:
            bool: Wether `name` is  available.
        """
        name = f"{uuid}/{name}"
        return self._is_marked(name)

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
    def _write(self, data: bytes, name: str) -> None:
        """Send the data `data` under the file name `name` without marking the file.

        This method should be implemented.

        Args:
            data (bytes): Data to send.
            name (str): Name of `data`.
        """

    @abc.abstractmethod
    def _read(self, name: str) -> bytes:
        """Read data from `name`.

        This function asserts that the file is ready and should be implemented by implementations.

        Args:
            name (str): Name to read.
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
    def to_spec(self) -> str:
        """Convert this channel to specification that can be serialised into JSON.

        Returns:
            dict[str, str]: Specification.
        """


class BlobStorageChannel(CommunicationChannel):
    """A communication channel via a folder in an Azure Blob Storage container."""

    def __init__(self, blob_folder: str) -> None:
        """Instantiate.

        Args:
            blob_folder (str): Folder to use. This must be a full URL that includes a SAS token
                with read and write permissions.
        """
        self.blob_folder = blob_folder
        if "?" not in blob_folder:
            raise ValueError("Given URL does not appear to contain a SAS token.")

    def to_spec(self) -> str:
        return self.blob_folder

    class Spec(BaseModel):
        class_name: Literal["BlobStorageChannel"]
        blob_folder: HttpUrl  # TODO: Can we force this to be `https`?

        def construct(self) -> "BlobStorageChannel":
            return BlobStorageChannel(blob_folder=str(self.blob_folder))

    def _blob_path(self, name: str) -> str:
        """For a given file name `name`, get the full path including the SAS token.

        Args:
            name (str): File name.

        Returns:
            str: Full path including the SAS token.
        """
        url, _, sas = self.blob_folder.partition("?")
        return f"{url.rstrip('/')}/{name.lstrip('/')}?{sas}"

    def _upload_blob(self, file_path: str, blob_name: str) -> None:
        """Upload a file to the blob."""
        blob_client = BlobClient.from_blob_url(self._blob_path(blob_name))
        with open(file_path, "rb") as f:
            blob_client.upload_blob(f, overwrite=True)

    def _download_blob(self, blob_name: str, file_path: str) -> None:
        """Download a blob to a file."""
        blob_client = BlobClient.from_blob_url(self._blob_path(blob_name))
        with open(file_path, "wb") as f:
            data = blob_client.download_blob().readall()
            f.write(data)

    def _send(self, batch: Batch, name: str) -> None:
        with tempfile.NamedTemporaryFile() as tf:
            tf.close()  # We will only use the name.
            batch.to_netcdf(tf.name)
            self._upload_blob(tf.name, name)

    def _receive(self, name: str) -> Batch:
        with tempfile.NamedTemporaryFile() as tf:
            tf.close()  # We will only use the name.
            self._download_blob(name, tf.name)
            return Batch.from_netcdf(tf.name)

    def _write(self, data: bytes, name: str) -> None:
        with tempfile.NamedTemporaryFile() as tf:
            tf.close()  # We will only use the name.
            with open(tf.name, "wb") as f:
                f.write(data)
            self._upload_blob(tf.name, name)

    def _read(self, name: str) -> bytes:
        with tempfile.NamedTemporaryFile() as tf:
            tf.close()  # We will only use the name.
            self._download_blob(name, tf.name)
            with open(tf.name, "rb") as f:
                return f.read()

    def _mark(self, name: str) -> None:
        with tempfile.TemporaryDirectory() as td:
            mark_file_path = Path(td) / f"{name}.finished"
            mark_file_path.parent.mkdir(exist_ok=True, parents=True)
            mark_file_path.write_text("File is available")
            self._upload_blob(str(mark_file_path), f"{name}.finished")

    def _is_marked(self, name: str) -> bool:
        res = requests.head(self._blob_path(f"{name}.finished"))
        return res.status_code == 200


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
