"""Copyright (c) Microsoft Corporation. Licensed under the MIT license."""

import abc
import logging
import os
from typing import Literal

import requests

__all__ = ["AbstractFoundryClient", "FoundryClient"]


logger = logging.getLogger(__name__)


class AbstractFoundryClient(metaclass=abc.ABCMeta):
    """A client to talk to Azure AI Foundry."""

    @abc.abstractmethod
    def score(self, data: dict) -> dict:
        """Send `data` to the scoring path.

        Args:
            data (dict): Data to send.

        Returns:
            dict: Answer.
        """


class FoundryClient(AbstractFoundryClient):
    def __init__(self, endpoint: str, token: str) -> None:
        """Initialise.

        Args:
            endpoint (str): URL to the endpoint.
            token (str): Authorisation token.
        """
        self.endpoint = endpoint
        self.token = token

    def _req(
        self,
        method: Literal["POST", "GET"],
        path: str,
        data: dict | None = None,
    ) -> requests.Response:
        return requests.request(
            method,
            os.path.join(self.endpoint, path),
            headers={
                "Authorization": f"Bearer {self.token}",
                "Content-Type": "application/json",
            },
            json=data,
        )

    def score(self, data: dict) -> dict:
        answer = self._req("POST", "score", {"data": data})
        answer.raise_for_status()
        return answer.json()
