"""Copyright (c) Microsoft Corporation. Licensed under the MIT license."""

import dataclasses
import math
from typing import Generator, Optional, Sequence

import torch

from aurora.batch import Batch
from aurora.model.aurora import Aurora
from aurora.model.swin3d import NoiseGenerator

__all__ = ["rollout"]


def _make_lead_time_tensor(batch: Batch, lead_time_hours: float) -> torch.Tensor:
    """Construct a per-sample lead-time tensor matching `batch`."""
    _example_variable = next(iter(batch.surf_vars.values()))
    return torch.full(
        (_example_variable.shape[0],),
        lead_time_hours,
        device=_example_variable.device,
        dtype=_example_variable.dtype,
    )


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
    apply_rollout_input_clipping: bool = True,
    generator: NoiseGenerator = None,
) -> Generator[Batch, None, None]:
    """Perform a roll-out to make long-term predictions.

    For Aurora models prior to Aurora 1.5, the rollout is straightforward: iteratively make a
    prediction based on inputs, then feed that prediction back as the new input for the next step.
    Aurora 1.5 introduces support for variable lead times, which enables sub-stepping within each
    main step. When `fine_lead_times` is provided, the model will produce predictions at each of the
    specified lead times within each main step, all of which are initialised from the same previous
    step and thus do not autoregress onto each other. We do require that the last `fine_lead_time`
    be the same as the model time step so we can construct proper next inputs. For instance, if
    specifying `fine_lead_times = [3, 6]` for a model with a 6-hour time step, the model will
    produce predictions at +3 and +6 hours, but only the +6 hour prediction will be fed back as
    input to continue the rollout.

    Aurora 1.5 Ensemble also introduces stochasticity. When `use_noise_accumulation` is `True`,
    noise will be continuously accumulated across both fine and major steps, introducing an auto-
    correlation between noise samples for more continuous predictions. Conveniently, setting the
    number of accumulation steps to the same length as `fine_lead_times` ensures the noise is
    effectively new between each major step, which matches the training regimen.

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
        apply_rollout_input_clipping (bool): Whether to apply the model's input clipping during
            roll-out. This is typically desirable to prevent unrealistic predictions from being fed
            back into the model during roll-out, but may be undesirable if the model was not trained
            with clipping and the user wants to preserve the raw model predictions for analysis.
            Default: `True`.
        generator (:class:`torch.Generator` or tuple of :class:`torch.Generator` or `None`,
            optional): Source of randomness for the noise injection of stochastic models, passed
            to every `forward` call of the roll-out. See :meth:`aurora.Aurora.forward` for the
            semantics. The generators advance on every (sub-)step, and a tuple corresponds to the
            order of the batch dimension throughout the roll-out. To reproduce a roll-out, re-seed
            the generators and call `model.reset_noise()` before starting. Default: `None`, which
            draws from the global RNG.

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
        if not math.isclose(fine_lead_times[-1], base_timestep_hours):
            raise ValueError(
                f"The last entry in `fine_lead_times` must equal the model's base time-step "
                f"of {base_timestep_hours} hours. Found {fine_lead_times[-1]} hours."
            )

    # Enable noise accumulation when the model is stochastic and sub-stepping.
    if use_noise_accumulation and fine_lead_times is not None:
        model.set_noise_accumulation(n=len(fine_lead_times))

    # Pre-compute the base lead-time tensor for models with variable lead time support.
    base_lead_times: Optional[torch.Tensor] = None
    if model.variable_lead_time:
        base_lead_times = _make_lead_time_tensor(batch, model.timestep.total_seconds() / 3600.0)

    for _ in range(steps):
        if fine_lead_times is not None:
            # Inner loop: iterate over sub-step lead times.
            for lt_hours in fine_lead_times:
                sub_lead_times = _make_lead_time_tensor(batch, lt_hours)
                pred = model.forward(batch, lead_times=sub_lead_times, generator=generator)

                yield pred

            # If desired, apply clipping before feeding predictions back as inputs.
            if apply_rollout_input_clipping:
                pred = model.apply_rollout_input_clipping(pred)
            batch = _advance_batch(batch, pred)
        else:
            pred = model.forward(batch, lead_times=base_lead_times, generator=generator)

            yield pred

            if apply_rollout_input_clipping:
                pred = model.apply_rollout_input_clipping(pred)
            batch = _advance_batch(batch, pred)

    # Disable noise accumulation after roll-out is complete, in case the model will be used for
    # normal inference or training afterwards.
    model.set_noise_accumulation(n=0)
