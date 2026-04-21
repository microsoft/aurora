"""Copyright (c) Microsoft Corporation. Licensed under the MIT license."""

import dataclasses
from typing import Generator, Optional, Sequence

import torch

from aurora.batch import Batch
from aurora.model.aurora import Aurora

__all__ = ["rollout"]


def _update_batch_lead_time(batch: Batch, lead_time_hours: float):
    """Return a copy of `batch` with the lead time updated to `lead_time_hours`."""
    _example_variable = next(iter(batch.surf_vars.values()))
    lead_time_tensor = torch.full(
        (_example_variable.shape[0],),
        lead_time_hours,
        device=_example_variable.device,
        dtype=_example_variable.dtype,
    )
    return dataclasses.replace(batch, lead_times=lead_time_tensor)


def _advance_batch(batch: Batch, pred: Batch) -> Batch:
    """Construct the next autoregressive input by sliding the history window.

    Removes the oldest time step and concatenating the new prediction. Only variables that are
    present in both the input batch and the prediction are concatenated, discarding output-only
    variables.
    """
    new_surf = {}
    for k, v in pred.surf_vars.items():
        if k in batch.surf_vars:
            new_surf[k] = torch.cat([batch.surf_vars[k][:, 1:], v], dim=1)
    new_atmos = {}
    for k, v in pred.atmos_vars.items():
        if k in batch.atmos_vars:
            new_atmos[k] = torch.cat([batch.atmos_vars[k][:, 1:], v], dim=1)
    return dataclasses.replace(pred, surf_vars=new_surf, atmos_vars=new_atmos)


def rollout(
    model: Aurora,
    batch: Batch,
    steps: int,
    fine_lead_times: Optional[Sequence[float]] = None,
    use_noise_accumulation: bool = True,
) -> Generator[Batch, None, None]:
    """Perform a roll-out to make long-term predictions.

    Args:
        model (:class:`aurora.Aurora`): The model to roll out.
        batch (:class:`aurora.Batch`): The batch to start the roll-out from.
        steps (int): The number of main roll-out steps.  Each step advances the
            forecast by the model's base time-step (typically 6 hours).
        fine_lead_times (sequence of float, optional): Sub-step lead times in hours to iterate
            within each main step. These sub-steps are all initialised from the previous main step
            and thus do not autoregress onto each other. For example, `[1, 2, 3, 4, 5, 6]` produces
            predictions at every hour. The *last* entry should equal the model's base time-step
            and is the one that advances the autoregressive state. Requires
            `model.variable_lead_time == True`. When `None` (default), no sub-stepping is performed
            and behaviour is unchanged from the original `rollout`.
        use_noise_accumulation (bool): Whether to enable noise accumulation when the model is
            stochastic and sub-stepping. This enables smoother transitions between `fine_lead_time`
            intermediate steps. It is intended to continue caching across fine and major steps to
            optimise smoothness across all lead times in the forecast. Has no effect when
            `fine_lead_times` is `None`. Default: `True`.

    Yields:
        :class:`aurora.Batch`: The prediction after every (sub-)step.
    """
    # We will need to concatenate data, so ensure that everything is already of the right form.
    batch = model.batch_transform_hook(batch)  # This might modify the available variables.
    # Use an arbitary parameter of the model to derive the data type and device.
    p = next(model.parameters())
    batch = batch.type(p.dtype)
    batch = batch.crop(model.patch_size)
    batch = batch.to(p.device)

    if fine_lead_times is not None and not model.variable_lead_time:
        raise ValueError("`fine_lead_times` requires `model.variable_lead_time=True`.")

    # Assert that the model's expected timestep is included at the end of `fine_lead_times`.
    if fine_lead_times is not None:
        base_timestep_hours = model.timestep.total_seconds() / 3600.0
        if fine_lead_times[-1] != base_timestep_hours:
            raise ValueError(
                f"The last entry in `fine_lead_times` must equal the model's base time-step "
                f"of {base_timestep_hours} hours. Found {fine_lead_times[-1]} hours."
            )

    # Enable noise accumulation when the model is stochastic and sub-stepping.
    if use_noise_accumulation and fine_lead_times is not None:
        model.set_noise_accumulation(True, n=len(fine_lead_times))

    # Pre-populate model lead times for models with variable lead time support.
    if model.variable_lead_time:
        batch = _update_batch_lead_time(batch, model.timestep.total_seconds() / 3600.0)

    for _ in range(steps):
        if fine_lead_times is not None:
            # Inner loop: iterate over sub-step lead times.
            for lt_hours in fine_lead_times:
                sub_batch = _update_batch_lead_time(batch, lt_hours)
                pred = model.forward(sub_batch)

                yield pred

            # Apply clipping before feeding predictions back as inputs.
            batch = _advance_batch(batch, model.apply_rollout_input_clipping(pred))
        else:
            pred = model.forward(batch)

            yield pred

            batch = _advance_batch(batch, model.apply_rollout_input_clipping(pred))

    # Disable noise accumulation after roll-out is complete, in case the model will be used for
    # normal inference or training afterwards.
    model.set_noise_accumulation(False)
