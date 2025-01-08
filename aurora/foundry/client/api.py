"""Copyright (c) Microsoft Corporation. Licensed under the MIT license.

This is the API that the end user uses to submit jobs to the model running on Azure AI Foundry.
"""

import logging
from typing import Generator, Literal, Optional, Union

from pydantic import BaseModel, Field

from aurora import Batch
from aurora.foundry.client.foundry import AbstractFoundryClient
from aurora.foundry.common.channel import CommunicationChannel, iterate_prediction_files
from aurora.foundry.common.model import models

__all__ = ["SubmissionError", "submit"]

logger = logging.getLogger(__name__)


class SubmissionInfo(BaseModel):
    kind: Literal["submission_info"]
    uuid: str


class ProgressInfo(BaseModel):
    kind: Literal["progress_info"]
    uuid: str
    completed: bool
    progress_percentage: int
    error: bool
    error_info: str


class Answer(BaseModel):
    success: bool
    message: str
    data: Optional[Union[SubmissionInfo, ProgressInfo]] = Field(..., discriminator="kind")


class SubmissionError(Exception):
    """The submission could not be completed for some reason."""


def submit(
    batch: Batch,
    model_name: str,
    num_steps: int,
    client_comm: CommunicationChannel,
    host_comm: CommunicationChannel,
    foundry_client: AbstractFoundryClient,
) -> Generator[Batch, None, None]:
    """Submit a request to Azure AI Foundry and retrieve the predictions.

    Args:
        batch (:class:`aurora.Batch`): Initial condition.
        model_name (str): Name of the model. This name must be available in
            :mod:`aurora_foundry.common.model`.
        num_steps (int): Number of prediction steps.
        client_comm (:class:`aurora_foundry.common.comm.CommunicationChannel`): Channel that the
            client uses to send and receive data.
        host_comm (:class:`aurora_foundry.common.comm.CommunicationChannel`): Channel that the host
            uses to send and receive data.
        foundry_client (:class:`aurora_foundry.client.foundry.AbstractFoundryClient`): Client to
            communicate with Azure Foundry AI.

    Yields:
        :class:`aurora.Batch`: Predictions.
    """
    if model_name not in models:
        raise KeyError(f"Model `{model_name}` is not a valid model.")

    # Send a request to the endpoint to produce the predictions.
    data = {
        "request": {
            "action": "submit",
            "model_name": model_name,
            "num_steps": num_steps,
            "host_comm": host_comm.to_spec(),
        }
    }
    answer = Answer(**foundry_client.score(data))
    if not answer.success:
        raise SubmissionError(answer.message)
    submission_info = answer.data
    if not isinstance(submission_info, SubmissionInfo):
        raise SubmissionError(
            "Server returned no submission information. "
            "Cannot determine task UUID to track tasks."
        )
    task_uuid = submission_info.uuid
    logger.info("Submitted request to endpoint.")

    # Send the initial condition over.
    client_comm.send(batch, task_uuid, "input.nc")

    previous_progress: int = 0

    while True:
        # Check on the progress of the task.
        data = {"request": {"action": "check", "uuid": task_uuid}}
        answer = Answer(**foundry_client.score(data))
        if not answer.success:
            raise SubmissionError(answer.message)
        progress_info = answer.data
        if not isinstance(progress_info, ProgressInfo):
            raise SubmissionError(
                "Server returned no progress information. "
                "Cannot determine whether the task has been completed or not."
            )

        if progress_info.error:
            raise SubmissionError(f"Task failed: {progress_info.error_info}")

        if progress_info.progress_percentage > previous_progress:
            logger.info(f"Task progress update: {progress_info.progress_percentage}%.")
            previous_progress = progress_info.progress_percentage

        if progress_info.completed:
            logger.info("Task has been completed!")
            break

    logger.info("Retrieving predictions.")
    for prediction_name in iterate_prediction_files("prediction.nc", num_steps):
        yield client_comm.receive(task_uuid, prediction_name)
