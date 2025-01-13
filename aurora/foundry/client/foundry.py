"""Copyright (c) Microsoft Corporation. Licensed under the MIT license."""

import logging
from typing import Literal

import requests

__all__ = ["FoundryClient"]


logger = logging.getLogger(__name__)


class FoundryClient:
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
            self.endpoint.rstrip("/") + "/" + path.lstrip("/"),
            headers={
                "Authorization": f"Bearer {self.token}",
                "Content-Type": "application/json",
            },
            json=data,
        )

    def submit_task(self, data: dict) -> dict:
        """Send `data` to the scoring path.

        Args:
            data (dict): Data to send.

        Returns:
            dict: Submission information.
        """
        answer = self._req("POST", "score", data)
        answer.raise_for_status()
        return answer.json()

    def get_progress(self, task_id: str) -> dict:
        """Get the progress of the task.

        Args:
            task_id (str): Task ID to get progress info for.

        Returns:
            dict: Progress information.
        """
        answer = self._req("GET", f"score?task_id={task_id}")
        answer.raise_for_status()
        return answer.json()
