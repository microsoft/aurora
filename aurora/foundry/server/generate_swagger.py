"""Copyright (c) Microsoft Corporation. Licensed under the MIT license."""

import json
import sys

from fastapi import Depends, FastAPI, HTTPException, Query, status
from fastapi.security import APIKeyHeader
from score import CreationResponse, Submission, TaskInfo

app = FastAPI(
    title="Aurora",
    description="Produce predictions with the Aurora model",
    version="1.0.0",
)
security = APIKeyHeader(
    name="Authorization", auto_error=False, description="Example 'Bearer myApiKey'"
)


async def get_api_key(api_key: str = Depends(security)):
    if api_key is None or not api_key.startswith("Bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key",
            headers={"WWW-Authenticate": "Bearer"},
        )
    token = api_key.split("Bearer ")
    # Here you can add your logic to validate the token
    if token != "your_actual_api_key":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return token


# POST method on /score endpoint
@app.post(
    "/score",
    response_model=CreationResponse,
    summary="Create a new task",
)
async def create_score(input: Submission, token: str = Depends(get_api_key)): ...


# GET method on /score endpoint
@app.get("/score", response_model=TaskInfo, summary="Update an existing task")
async def get_score(
    task_id: str = Query(
        ...,
        description="The ID of the task",
        examples=dict(task_id=dict(value="abc-123-def", summary="Sample Task ID")),
    ),
    token: str = Depends(get_api_key),
): ...


# Liveness route
@app.get("/", summary="Succeeds when the service is ready.")
async def read_root():
    return "Healthy"


# Route to get the Swagger file
@app.get("/swagger.json")
async def get_swagger():
    return app.openapi()


def dump_openapi_spec(fn):
    openapi_spec = app.openapi()
    with open(fn, "w") as f:
        json.dump(openapi_spec, f, indent=2)
    print(f"OpenAPI spec dumped to {fn}")


if __name__ == "__main__":
    if not sys.argv[1:]:
        import uvicorn

        uvicorn.run(app, host="0.0.0.0", port=8000)
    else:
        dump_openapi_spec(sys.argv[1])
