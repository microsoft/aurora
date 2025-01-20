"""Copyright (c) Microsoft Corporation. Licensed under the MIT license."""

import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from uuid import uuid4

import mlflow.pyfunc
from pydantic import BaseModel, HttpUrl

from aurora.foundry.common.channel import (
    BlobStorageChannel,
    iterate_prediction_files,
)
from aurora.foundry.common.model import MLFLOW_ARTIFACTS, models

__all__ = ["AuroraModelWrapper"]

# Need to give the name explicitly here, because the script may be run stand-alone.
logger = logging.getLogger("aurora.foundry.server.score")


class Submission(BaseModel):
    data_folder_uri: HttpUrl
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


class CreationResponse(BaseModel):
    task_id: str

    class Config:
        json_schema_extra = dict(example=dict(task_id="abc-123-def"))


class TaskInfo(BaseModel):
    task_id: str
    completed: bool
    progress_percentage: int
    success: bool | None
    submitted: bool
    status: str

    class Config:
        json_schema_extra = dict(
            example=dict(
                task_id="abc-123-def",
                completed=True,
                progress_percentage=100,
                success=None,
                submitted=True,
                error_info="Queued",
            )
        )


class Task:
    def __init__(self, submission: Submission):
        self.submission: Submission = submission

        self.task_info = TaskInfo(
            # TODO: Make sure that this `uuid` really is unique!
            task_id=str(uuid4()),
            completed=False,
            progress_percentage=0,
            success=None,
            submitted=False,
            status="Unsubmitted",
        )

    def __call__(self) -> None:
        self.task_info.status = "Running"

        try:
            submission = self.submission
            channel = BlobStorageChannel(str(submission.data_folder_uri))

            model_class = models[submission.model_name]
            model = model_class()

            batch = channel.receive(self.task_info.task_id, "input.nc")

            logger.info("Running predictions.")
            for i, (pred, path) in enumerate(
                zip(
                    model.run(batch, submission.num_steps),
                    iterate_prediction_files("prediction.nc", submission.num_steps),
                )
            ):
                channel.send(pred, self.task_info.task_id, path)

                self.task_info.progress_percentage = int((100 * (i + 1)) / submission.num_steps)

            self.task_info.success = True
            self.task_info.status = "Successfully completed"

        except Exception as exc:
            self.task_info.success = False
            self.task_info.status = f"Exception: {str(exc)}"

        finally:
            self.task_info.completed = True


class AuroraModelWrapper(mlflow.pyfunc.PythonModel):
    """A wrapper around an async workflow for making predictions with Aurora."""

    def load_context(self, context) -> None:
        logging.getLogger("aurora").setLevel(logging.INFO)
        logger.info("Starting `ThreadPoolExecutor`.")
        self.POOL = ThreadPoolExecutor(max_workers=1)
        self.TASKS: dict[str, Task] = {}
        self.POOL.__enter__()
        MLFLOW_ARTIFACTS.update(context.artifacts)

    def predict(self, context, model_input: dict, params=None) -> dict:
        data = json.loads(model_input["data"].item())

        if data["type"] == "submission":
            logger.info("Creating a new task.")
            task = Task(Submission(**data["msg"]))
            self.TASKS[task.task_info.task_id] = task
            return CreationResponse(task_id=task.task_info.task_id).dict()

        elif data["type"] == "task_info":
            logger.info("Processing an existing task.")

            task_id = data["msg"]["task_id"]
            if not task_id:
                raise Exception("Missing `task_id` parameter.")
            if task_id not in self.TASKS:
                raise Exception("Task ID cannot be found.")

            task = self.TASKS[task_id]

            if not task.task_info.submitted:
                # Attempt to submit the task if the initial condition is available.

                channel = BlobStorageChannel(str(task.submission.data_folder_uri))
                if channel.exists(task_id, "input.nc"):
                    logger.info("Initial condition was found. Submitting task.")
                    # Send an acknowledgement back to test that the host can write. The client will
                    # check for this acknowledgement.
                    channel.write(b"Acknowledgement of initial condition", task_id, "input.nc.ack")

                    # Queue the task.
                    task.task_info.submitted = True
                    task.task_info.status = "Queued"
                    self.POOL.submit(task)

                else:
                    logger.info("Initial condition not available. Waiting.")

                    # Wait a little to prevent the client for querying too frequently.
                    time.sleep(3)

            else:
                logger.info("Task still running. Waiting.")

                # Wait a little to prevent the client for querying too frequently. While waiting,
                # do check for the task to be completed.
                for _ in range(3):
                    if task.task_info.completed:
                        break
                    time.sleep(1)

            return task.task_info.dict()

        else:
            raise ValueError(f"Unknown data type: `{data['type']}`.")
