"""Copyright (c) Microsoft Corporation. Licensed under the MIT license.

This is the API that the end user uses to submit jobs to the model running on Azure AI Foundry.
"""

import logging
from typing import Generator

from pydantic import BaseModel

from aurora import Batch
from aurora.foundry.client.foundry import AbstractFoundryClient
from aurora.foundry.common.channel import CommunicationChannel, iterate_prediction_files
from aurora.foundry.common.model import models

__all__ = ["SubmissionError", "submit"]

logger = logging.getLogger(__name__)


class SubmissionInfo(BaseModel):
    task_id: str


class ProgressInfo(BaseModel):
    task_id: str
    completed: bool
    progress_percentage: int
    error: bool
    error_info: str


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
    task = {
        "model_name": model_name,
        "num_steps": num_steps,
        "data_folder_uri": host_comm.to_spec(),
    }
    response = foundry_client.submit_task(task)
    try:
        submission_info = SubmissionInfo(**response)
    except Exception as e:
        raise SubmissionError(response["message"]) from e
    task_id = submission_info.task_id
    logger.info("Submitted task %r to endpoint.", task_id)

    # Send the initial condition over.
    client_comm.send(batch, task_id, "input.nc")

    previous_progress: int = 0

    while True:
        # Check on the progress of the task.
        response = foundry_client.get_progress(task_id)
        progress_info = ProgressInfo(**response)

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
        yield client_comm.receive(task_id, prediction_name)
