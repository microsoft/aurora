"""Copyright (c) Microsoft Corporation. Licensed under the MIT license.

This is the API that the end user uses to submit jobs to the model running on Azure AI Foundry.
"""

import logging
from typing import Generator

from pydantic import BaseModel

from aurora import Batch
from aurora.foundry.client.foundry import FoundryClient
from aurora.foundry.common.channel import CommunicationChannel, iterate_prediction_files
from aurora.foundry.common.model import models

__all__ = ["SubmissionError", "submit"]

logger = logging.getLogger(__name__)


class CreationInfo(BaseModel):
    task_id: str


class TaskInfo(BaseModel):
    task_id: str
    completed: bool
    progress_percentage: int
    success: bool | None
    submitted: bool
    status: str


class SubmissionError(Exception):
    """The submission could not be completed for some reason."""


def submit(
    batch: Batch,
    model_name: str,
    num_steps: int,
    channel: CommunicationChannel,
    foundry_client: FoundryClient,
) -> Generator[Batch, None, None]:
    """Submit a request to Azure AI Foundry and retrieve the predictions.

    Args:
        batch (:class:`aurora.Batch`): Initial condition.
        model_name (str): Name of the model. This name must be available in
            :mod:`aurora.foundry.common.model`. See the Aurora Foundry Python API documentation for
             which models are available.
        num_steps (int): Number of prediction steps.
        channel (:class:`aurora.foundry.common.channel.CommunicationChannel`): Channel to use for
            sending and receiving data.
        foundry_client (:class:`aurora.foundry.client.foundry.FoundryClient`): Client to
            communicate with Azure Foundry AI.

    Yields:
        :class:`aurora.Batch`: Predictions.
    """
    if model_name not in models:
        raise KeyError(f"Model `{model_name}` is not a valid model.")

    # Create a task at the endpoint.
    task = {
        "model_name": model_name,
        "num_steps": num_steps,
        "data_folder_uri": channel.to_spec(),
    }
    response = foundry_client.submit_task(task)
    try:
        submission_info = CreationInfo(**response)
    except Exception as e:
        raise SubmissionError("Failed to create task.") from e
    task_id = submission_info.task_id
    logger.info(f"Created task `{task_id}` at endpoint.")

    # Send the initial condition over.
    logger.info("Uploading initial condition.")
    channel.send(batch, task_id, "input.nc")

    previous_status: str = "No status"
    previous_progress: int = 0
    ack_read: bool = False

    while True:
        # Check on the progress of the task. The first progress check will trigger the task to be
        # submitted.
        response = foundry_client.get_progress(task_id)
        task_info = TaskInfo(**response)

        if task_info.submitted and not ack_read:
            # If the task has been submitted, we must be able to read the acknowledgement of the
            # initial condition.
            try:
                channel.read(task_id, "input.nc.ack", timeout=120)
                ack_read = True  # Read the acknowledgement only once.
            except TimeoutError as e:
                raise SubmissionError(
                    "Could not read acknowledgement of initial condition. "
                    "This acknowledgement should be availabe, "
                    "since the task has been successfully submitted. "
                    "Something might have gone wrong in the communication "
                    "between the client and the server. "
                    "Please check the logs and your SAS token should you be using one."
                ) from e

        if task_info.status != previous_status:
            logger.info(f"Task status update: {task_info.status}")
            previous_status = task_info.status

        if task_info.progress_percentage > previous_progress:
            logger.info(f"Task progress update: {task_info.progress_percentage}%.")
            previous_progress = task_info.progress_percentage

        if task_info.completed:
            if task_info.success:
                logger.info("Task has been successfully completed!")
                break
            else:
                raise SubmissionError(f"Task failed: {task_info.status}")

    logger.info("Retrieving predictions.")
    for prediction_name in iterate_prediction_files("prediction.nc", num_steps):
        yield channel.receive(task_id, prediction_name)
    logger.info("All predictions have been retrieved.")
