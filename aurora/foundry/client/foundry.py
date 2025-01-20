"""Copyright (c) Microsoft Corporation. Licensed under the MIT license."""

import json
import logging

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
        data: dict | None = None,
    ) -> requests.Response:
        wrapped = {"data": json.dumps(data)}
        return requests.request(
            "POST",
            self.endpoint,
            headers={
                "Authorization": f"Bearer {self.token}",
                "Content-Type": "application/json",
            },
            json={"input_data": wrapped},
        )

    def _unwrap(self, response: requests.Response) -> dict:
        if not response.ok:
            logger.error(response.text)
        response.raise_for_status()
        response_json = response.json()
        return response_json

    def submit_task(self, data: dict) -> dict:
        """Send `data` to the scoring path.

        Args:
            data (dict): Data to send.

        Returns:
            dict: Submission information.
        """
        answer = self._req({"type": "submission", "msg": data})
        return self._unwrap(answer)

    def get_progress(self, task_id: str) -> dict:
        """Get the progress of the task.

        Args:
            task_id (str): Task ID to get progress info for.

        Returns:
            dict: Progress information.
        """
        answer = self._req({"type": "task_info", "msg": {"task_id": task_id}})
        return self._unwrap(answer)
