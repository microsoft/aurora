"""Copyright (c) Microsoft Corporation. Licensed under the MIT license."""

import abc
import logging
from typing import Generator

import torch

import aurora
from aurora import Batch, rollout

__all__ = ["Model", "models"]

logger = logging.getLogger(__name__)

# A dictionary containing ``<name, artifact_path>`` entries, where ``artifact_path`` is an
# absolute filesystem path to the artifact.
MLFLOW_ARTIFACTS: dict[str, str] = dict()


class Model(metaclass=abc.ABCMeta):
    """A model that can run predictions."""

    def __init__(self):
        """Initialise.

        This creates and loads the model and determines the device the run the model on.
        """
        self.model = self.create_model()
        self.model.eval()

        if torch.cuda.is_available():
            logger.info("GPU detected. Running on GPU.")
            self.target_device = torch.device("cuda")
        else:
            logger.warning("No GPU available. Running on CPU.")
            self.target_device = torch.device("cpu")

    @abc.abstractmethod
    def create_model(self) -> aurora.Aurora:
        """Create the model.

        Returns:
            :class:`aurora.Aurora`: Model.
        """

    @torch.inference_mode
    def run(self, batch: Batch, num_steps: int) -> Generator[Batch, None, None]:
        """Perform predictions on the target device.

        Args:
            batch (:class:`aurora.Batch`): Initial condition.
            num_steps (int): Number of prediction steps.

        Returns:
            :class:`aurora.Aurora`: Model.
        """
        # Move batch and model to target device.
        self.model.to(self.target_device)  # Modifies in-place!
        batch = batch.to(self.target_device)

        # Perform predictions, immediately moving the output to the CPU.
        for pred in rollout(self.model, batch, steps=num_steps):
            yield pred.to("cpu")

        # Move batch and model back to the CPU.
        batch = batch.to("cpu")
        self.model.cpu()  # Modifies in-place!


class AuroraFineTuned(Model):
    name = "aurora-0.25-finetuned"
    """str: Name of the model."""

    def create_model(self) -> aurora.Aurora:
        model = aurora.Aurora()
        model.load_checkpoint_local(MLFLOW_ARTIFACTS[self.name])
        return model


class AuroraPretrained(Model):
    name = "aurora-0.25-pretrained"
    """str: Name of the model."""

    def create_model(self) -> aurora.Aurora:
        model = aurora.AuroraPretrained()
        model.load_checkpoint_local(MLFLOW_ARTIFACTS[self.name])
        return model


class AuroraSmallPretrained(Model):
    name = "aurora-0.25-small-pretrained"
    """str: Name of the model."""

    def create_model(self) -> aurora.Aurora:
        model = aurora.AuroraSmallPretrained()
        model.load_checkpoint_local(MLFLOW_ARTIFACTS[self.name])
        return model


class Aurora12hPretrained(Model):
    name = "aurora-0.25-12h-pretrained"
    """str: Name of the model."""

    def create_model(self) -> aurora.Aurora:
        model = aurora.Aurora12hPretrained()
        model.load_checkpoint_local(MLFLOW_ARTIFACTS[self.name])
        return model


class AuroraHighRes(Model):
    name = "aurora-0.1-finetuned"
    """str: Name of the model."""

    def create_model(self) -> aurora.Aurora:
        model = aurora.AuroraHighRes()
        model.load_checkpoint_local(MLFLOW_ARTIFACTS[self.name])
        return model


class AuroraAirPollution(Model):
    name = "aurora-0.4-air-pollution"
    """str: Name of the model."""

    def create_model(self) -> aurora.Aurora:
        model = aurora.AuroraAirPollution()
        model.load_checkpoint_local(MLFLOW_ARTIFACTS[self.name])
        return model


models: dict[str, type[Model]] = {}
"""dict[str, type[Model]]: A dictionary that lists all available models by their name."""

for model_class in Model.__subclasses__():
    assert hasattr(model_class, "name"), f"`{model_class.__name__}` is missing `name`."
    # `mypy` will complain, because `Model` is abstract, so it cannot be passed to `type`.
    models[model_class.name] = model_class  # type: ignore[type-abstract]
