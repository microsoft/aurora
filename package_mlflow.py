"""Copyright (c) Microsoft Corporation. Licensed under the MIT license.

Package the model with MLflow.
"""

from pathlib import Path

import mlflow.pyfunc
from huggingface_hub import hf_hub_download

from aurora.foundry.common.model import models
from aurora.foundry.server.mlflow_wrapper import AuroraModelWrapper

artifacts: dict[str, str] = {}

# Download all checkpoints into a local directory which will be included in the package.
ckpt_dir = Path("checkpoints")
ckpt_dir.mkdir(parents=True, exist_ok=True)
for name in models:
    hf_hub_download(
        repo_id="microsoft/aurora",
        filename=f"{name}.ckpt",
        local_dir=ckpt_dir,
    )
    artifacts = {name: str(ckpt_dir / f"{name}.ckpt")}


mlflow_pyfunc_model_path = "./aurora_mlflow_pyfunc"
mlflow.pyfunc.save_model(
    path=mlflow_pyfunc_model_path,
    code_paths=["aurora"],
    python_model=AuroraModelWrapper(),
    artifacts=artifacts,
    conda_env={
        "name": "aurora-mlflow-env",
        "channels": ["conda-forge"],
        "dependencies": [
            "python=3.11.11",
            "pip<=24.3.1",
            {
                "pip": [
                    "mlflow==2.19.0",
                    "cloudpickle==3.1.0",
                    "defusedxml==0.7.1",
                    "einops==0.8.0",
                    "jaraco-collections==5.1.0",
                    "numpy==2.2.1",
                    "scipy==1.15.1",
                    "timm==0.6.13",
                    "torch==2.5.1",
                    "torchvision==0.20.1",
                    "huggingface-hub==0.27.1",
                    "pydantic==2.10.5",
                    "xarray==2025.1.1",
                    "netCDF4==1.7.2",
                    "azure-storage-blob==12.24.0",
                ],
            },
        ],
    },
)
