"""Copyright (c) Microsoft Corporation. Licensed under the MIT license."""

import logging
import time
from concurrent.futures import ThreadPoolExecutor
from uuid import uuid4

try:
    from azureml_inference_server_http.api.aml_request import AMLRequest, rawhttp
except ImportError:
    # It's OK if this is not available.

    AMLRequest = dict

    def rawhttp(x):
        return x


from pydantic import BaseModel, HttpUrl

import aurora.foundry.server._hook  # noqa: F401
from aurora.foundry.common.channel import (
    BlobStorageChannel,
    iterate_prediction_files,
)
from aurora.foundry.common.model import models

__all__ = ["init", "run"]

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


POOL = ThreadPoolExecutor(max_workers=1)
TASKS = dict()


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


def init() -> None:
    """Initialise."""
    # Do not load the model here, because which model we need depends on the submission.
    logging.getLogger("aurora").setLevel(logging.INFO)
    logger.info("Starting ThreadPoolExecutor")
    POOL.__enter__()


@rawhttp
def run(input_data: AMLRequest) -> dict:
    """Perform predictions.

    Args:
        input_data (AMLRequest): Mostly a Flask `Request` object.

    Returns:
        dict: The response to the request. This `dict` is implicitly a status 200 `AMLResponse`.
    """
    logger.info("Received request.")

    if input_data.method == "POST":
        logger.info("Creating a new task.")
        task = Task(Submission(**input_data.get_json()))
        TASKS[task.task_info.task_id] = task
        return CreationResponse(task_id=task.task_info.task_id).dict()

    elif input_data.method == "GET":
        logger.info("Processing an existing task.")

        task_id = input_data.args.get("task_id")
        if not task_id:
            raise Exception("Missing `task_id` query parameter.")
        if task_id not in TASKS:
            raise Exception("Task ID cannot be found.")

        task = TASKS[task_id]

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
                POOL.submit(task)

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

    raise Exception("Method not allowed.")  # This branch should be unreachable.
