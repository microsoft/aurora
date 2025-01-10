"""Copyright (c) Microsoft Corporation. Licensed under the MIT license."""

import logging
import time
from concurrent.futures import ThreadPoolExecutor
from uuid import uuid4

from azureml_inference_server_http.api.aml_request import AMLRequest, rawhttp
from pydantic import BaseModel

import aurora.foundry.server._hook  # noqa: F401
from aurora.foundry.common.channel import (
    BlobStorageCommunication,
    iterate_prediction_files,
)
from aurora.foundry.common.model import models

__all__ = ["init", "run"]

# Need to give the name explicitly here, because the script may be run stand-alone.
logger = logging.getLogger("aurora.foundry.server.score")


class Submission(BaseModel):
    data_folder_uri: str
    model_name: str
    num_steps: int

    class Config:
        json_schema_extra = dict(
            example=dict(
                data_folder_uri="https://my.blob.core.windows.net/container/some/path?WRITABLE_SAS",
                model_name="aurora-0.25-small-pretrained",
                num_steps=5,
            )
        )


class SubmissionResponse(BaseModel):
    task_id: str

    class Config:
        json_schema_extra = dict(example=dict(task_id="abc-123-def"))


class ProgressInfo(BaseModel):
    task_id: str
    completed: bool
    progress_percentage: int
    error: bool
    error_info: str

    class Config:
        json_schema_extra = dict(
            example=dict(
                task_id="abc-123-def",
                completed=True,
                progress_percentage=100,
                error=False,
                error_info="",
            )
        )


POOL = ThreadPoolExecutor(max_workers=1)
TASKS = dict()


class Task:
    def __init__(self, request: Submission):
        self.request: Submission = request
        # TODO: Make sure that this `uuid` really is unique!
        self.uuid: str = str(uuid4())
        self.progress_percentage: int = 0
        self.completed: bool = False
        self.exc: Exception | None = None

    def __call__(self) -> None:
        try:
            request = self.request
            host_comm = BlobStorageCommunication(request.data_folder_uri)

            model_class = models[request.model_name]
            model = model_class()

            batch = host_comm.receive(self.uuid, "input.nc")

            logger.info("Running predictions.")
            for i, (pred, path) in enumerate(
                zip(
                    model.run(batch, request.num_steps),
                    iterate_prediction_files("prediction.nc", request.num_steps),
                )
            ):
                host_comm.send(pred, self.uuid, path)
                self.progress_percentage = int((100 * (i + 1)) / request.num_steps)

            self.completed = True

        except Exception as exc:
            self.exc = exc


def init() -> None:
    """Initialise. Do not load the model here, because which model we need depends on the
    request."""
    POOL.__enter__()


@rawhttp
def run(input_data: AMLRequest) -> dict:
    """Perform predictions.

    Args:
        input_data (AMLRequest): Mostly a Flask Request object.

    Returns:
        dict/AMLResponse: The response to the request. dicts are implictitly 200 AMLResponses.
    """
    logger.info("Received request.")

    if input_data.method == "POST":
        logger.info("Submitting new task to thread pool.")
        task = Task(Submission(**input_data.get_json()))
        POOL.submit(task)
        TASKS[task.uuid] = task
        return SubmissionResponse(task_id=task.uuid).dict()

    elif input_data.method == "GET":
        logger.info("Returning the status of an existing task.")
        task_id = input_data.args.get("task_id")
        if not task_id:
            raise Exception("Missing `task_id` query parameter.")
        if task_id not in TASKS:
            raise Exception("Task ID cannot be found.")
        else:
            task = TASKS[task_id]
            # Allow the task some time to complete.
            # We sleep here so the client does not query too frequently.
            for _ in range(3):
                if task.completed:
                    break
                time.sleep(1)

            return ProgressInfo(
                task_id=task_id,
                completed=task.completed,
                progress_percentage=task.progress_percentage,
                error=task.exc is not None,
                error_info=str(task.exc) if task.exc else "",
            ).dict()

    raise Exception("Method not allowed.")  # This branch should be unreachable.
