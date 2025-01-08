"""Copyright (c) Microsoft Corporation. Licensed under the MIT license."""

import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Literal, Union
from uuid import uuid4

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
    action: Literal["submit"]
    host_comm: CommSpecs = Field(..., discriminator="class_name")
    model_name: str
    num_steps: int


class Check(BaseModel):
    action: Literal["check"]
    uuid: str


class Request(BaseModel):
    request: Union[Submission, Check] = Field(..., discriminator="action")


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


def run(raw_data: str) -> dict:
    """Perform predictions.

    Args:
        raw_data (str): Request as JSON.

    Returns:
        dict: Answer, which will be encoded as JSON.
    """
    logger.info("Received request.")
    raw = json.loads(raw_data)
    request = Request(**raw["data"]).request

    if isinstance(request, Submission):
        logger.info("Submitting new task to thread pool.")
        task = Task(request)
        POOL.submit(task)
        TASKS[task.uuid] = task
        return {
            "success": True,
            "message": "Request has been succesfully submitted.",
            "data": {
                "kind": "submission_info",
                "uuid": task.uuid,
            },
        }

    elif isinstance(request, Check):
        logger.info("Returning the status of an existing task.")
        uuid = request.uuid
        time.sleep(1)  # Sleep here, so the client does not need to.
        if uuid not in TASKS:
            return {
                "success": False,
                "message": "Task UUID cannot be found.",
            }
        else:
            task = TASKS[uuid]
            return {
                "success": True,
                "message": "Status check completed.",
                "data": {
                    "kind": "progress_info",
                    "uuid": uuid,
                    "completed": task.completed,
                    "progress_percentage": task.progress_percentage,
                    "error": task.exc is not None,
                    "error_info": str(task.exc) if task.exc else "",
                },
            }

    else:
        # This branch should be unreachable.
        return {
            "success": False,
            "message": "Invalid action.",
        }
