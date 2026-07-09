"""Copyright (c) Microsoft Corporation. Licensed under the MIT license.

This is the API that the end user uses to submit jobs to the model running on Azure AI Foundry.
"""

import logging
import warnings
from typing import Generator, Optional, Sequence, Union

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
    fine_lead_times: Optional[Sequence[float]] = None,
    saved_surf_vars: Optional[tuple[str, ...]] = None,
    saved_atmos_vars: Optional[tuple[str, ...]] = None,
    saved_atmos_levels: Optional[tuple[int | float, ...]] = None,
    saved_static_vars: Optional[tuple[str, ...]] = None,
    async_upload_workers: int = 0,
    return_urls: bool = False,
) -> Generator[Union[Batch, str], None, None]:
    """Submit a request to Azure AI Foundry and retrieve the predictions.

    Args:
        batch (:class:`aurora.Batch`): Initial condition.
        model_name (str): Name of the model. This name must be available in
            :mod:`aurora.foundry.common.model`. See the Aurora Foundry Python API documentation for
            which models are available.
        num_steps (int): Number of main prediction steps.
        channel (:class:`aurora.foundry.common.channel.CommunicationChannel`): Channel to use for
            sending and receiving data.
        foundry_client (:class:`aurora.foundry.client.foundry.FoundryClient`): Client to
            communicate with Azure Foundry AI.
        fine_lead_times (sequence of float, optional): Sub-step lead times in hours within each main
            step.  When provided, the total number of predictions returned is
            `num_steps * len(fine_lead_times)`. See :func:`aurora.rollout` for details.
            Aurora-1.5 only.
        saved_surf_vars (tuple[str, ...], optional): Surface variables to keep in the saved
            predictions. `None` (default) keeps all variables. An empty tuple removes all surface
            variables from the output. Aurora-1.5 only.
        saved_atmos_vars (tuple[str, ...], optional): Atmospheric variables to keep in the saved
            predictions. `None` keeps all. An empty tuple removes all. Aurora-1.5 only.
        saved_atmos_levels (tuple[int | float, ...], optional): Pressure levels (hPa) to keep in
            the saved predictions. `None` keeps all levels. An empty tuple removes all levels.
            Aurora-1.5 only.
        saved_static_vars (tuple[str, ...], optional): Static variables to keep in the saved
            predictions. `None` keeps all. An empty tuple removes all. Aurora-1.5 only.
        async_upload_workers (int, optional): If > 0, the server will serialize and upload
            predictions in parallel worker processes so that GPU inference can proceed concurrently.
            The value controls the number of worker processes. This can speed up end-to-end
            prediction time significantly. Values > 0 are only valid for the Aurora-1.5 Foundry
            model.
        return_urls (bool, optional): If `True`, yield the blob storage URLs of the predictions
            instead of downloading them. This is useful to avoid downloading and opening large files
            on the local client, instead providing the URLs for later download. Defaults to `False`.

    Yields:
        :class:`aurora.Batch`: Predictions.
    """
    if model_name not in models:
        raise KeyError(f"Model `{model_name}` is not a valid model.")

    # Warn about features that require the new Aurora-1.5 Foundry model deployment.
    is_aurora_1p5 = model_name in ["aurora-0.25-v1.5", "aurora-0.25-v1.5-ensemble"]
    aurora_1p5_features: list[str] = []
    if fine_lead_times is not None:
        aurora_1p5_features.append("fine_lead_times")
    if saved_surf_vars is not None:
        aurora_1p5_features.append("saved_surf_vars")
    if saved_atmos_vars is not None:
        aurora_1p5_features.append("saved_atmos_vars")
    if saved_atmos_levels is not None:
        aurora_1p5_features.append("saved_atmos_levels")
    if saved_static_vars is not None:
        aurora_1p5_features.append("saved_static_vars")
    if aurora_1p5_features and not is_aurora_1p5:
        warnings.warn(
            f"You are using a model ('{model_name}') which is not an Aurora-1.5 model. The "
            f"following parameters only work on a deployment of Aurora-1.5 and will therefore be "
            f"silently ignored by the Aurora endpoint: {', '.join(aurora_1p5_features)}.",
            stacklevel=2,
        )

    # Create a task at the endpoint.
    task: dict = {
        "model_name": model_name,
        "num_steps": num_steps,
        "data_folder_uri": channel.to_spec(),
    }
    if fine_lead_times is not None:
        task["fine_lead_times"] = list(fine_lead_times)
    if saved_surf_vars is not None:
        task["saved_surf_vars"] = list(saved_surf_vars)
    if saved_atmos_vars is not None:
        task["saved_atmos_vars"] = list(saved_atmos_vars)
    if saved_atmos_levels is not None:
        task["saved_atmos_levels"] = list(saved_atmos_levels)
    if saved_static_vars is not None:
        task["saved_static_vars"] = list(saved_static_vars)
    task["async_upload_workers"] = async_upload_workers
    response = foundry_client.submit_task(task)
    try:
        submission_info = CreationInfo(**response)
    except Exception as e:
        raise SubmissionError(f"Failed to create task. Endpoint response: {response}") from e
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

    n_fine = len(fine_lead_times) if fine_lead_times else 1
    total_predictions = num_steps * n_fine

    pred_file = "prediction.nc"
    if return_urls:
        logger.info("Retrieving prediction URLs.")
        for prediction_name in iterate_prediction_files(pred_file, total_predictions):
            yield channel.get_url(task_id, prediction_name)
        logger.info("All prediction URLs have been retrieved.")
    else:
        logger.info("Retrieving predictions.")
        for prediction_name in iterate_prediction_files(pred_file, total_predictions):
            yield channel.receive(task_id, prediction_name)
        logger.info("All predictions have been retrieved.")
