"""Copyright (c) Microsoft Corporation. Licensed under the MIT license."""

import logging
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Literal, Union
from uuid import uuid4

from azureml_inference_server_http.api.aml_request import AMLRequest, rawhttp
from azureml_inference_server_http.api.aml_response import AMLResponse
from pydantic import BaseModel, Field

from aurora.foundry.common.channel import (
    BlobStorageCommunication,
    LocalCommunication,
    iterate_prediction_files,
)
from aurora.foundry.common.model import models

__all__ = ["init", "run"]

# Need to give the name explicitly here, because the script may be run stand-alone.
logger = logging.getLogger("aurora.foundry.server.score")

CommSpecs = Union[LocalCommunication.Spec, BlobStorageCommunication.Spec]


class Submission(BaseModel):
    host_comm: CommSpecs = Field(..., discriminator="class_name")
    model_name: str
    num_steps: int


class Check(BaseModel):
    action: Literal["check"]
    uuid: str


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
            host_comm = request.host_comm.construct()

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
        try:
            task = Task(Submission(**input_data.get_json()))
        except Exception as exc:
            return AMLResponse(dict(message=str(exc)), 500, {}, json_str=True)
        POOL.submit(task)
        TASKS[task.uuid] = task
        return {
            "task_id": task.uuid,
        }

    elif input_data.method == "GET":
        logger.info("Returning the status of an existing task.")
        uuid = input_data.args.get("task_id")
        if not uuid:
            return AMLResponse(
                dict(message="Missing task_id query parameter."),
                400,
                {},
                json_str=True,
            )
        if uuid not in TASKS:
            return {
                "success": False,
                "message": "Task UUID cannot be found.",
            }
        else:
            task = TASKS[uuid]
            # Allow the task some time to complete.
            # We sleep here so the client does not query too frequently.
            for _ in range(3):
                if task.completed:
                    break
                time.sleep(1)
            return {
                "task_id": uuid,
                "completed": task.completed,
                "progress_percentage": task.progress_percentage,
                "error": task.exc is not None,
                "error_info": str(task.exc) if task.exc else "",
            }

    else:
        # This branch should be unreachable.
        return AMLResponse(dict(message="Method not allowed."), 405, {}, json_str=True)
