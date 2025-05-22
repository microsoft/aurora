"""Copyright (c) Microsoft Corporation. Licensed under the MIT license."""

from datetime import datetime, timedelta

import numpy as np
import torch

from aurora import AuroraSmallPretrained, Batch, Metadata, rollout


def test_rollout():
    # Construct two models which are initialised exactly the same, but the one uses a separate
    # LoRA for every step and the other does not.
    model1 = AuroraSmallPretrained(use_lora=True, lora_mode="single")
    model1.load_checkpoint(
        "microsoft/aurora",
        "aurora-0.25-small-pretrained.ckpt",
        strict=False,  # LoRA parameters not available.
    )
    model2 = AuroraSmallPretrained(use_lora=True, lora_mode="all")

    sd1 = model1.state_dict()
    sd2 = model2.state_dict()
    for k in sd2:
        if k in sd1:
            # The Bs of the LoRAs are initalised to zero. We need to change that.
            if "lora_B" in k:
                torch.nn.init.kaiming_uniform_(sd1[k])
            # Copy the init for `model1`.
            sd2[k] = sd1[k]
        elif "lora_B" in k:
            # Not present in `model1`, so randomly init.
            torch.nn.init.kaiming_uniform_(sd2[k])
    model1.load_state_dict(sd1)
    model2.load_state_dict(sd2)

    # Disable drop-out.
    model1.eval()
    model2.eval()

    batch = Batch(
        surf_vars={k: torch.randn(1, 2, 17, 32) for k in ("2t", "10u", "10v", "msl")},
        static_vars={k: torch.randn(17, 32) for k in ("lsm", "z", "slt")},
        atmos_vars={k: torch.randn(1, 2, 4, 17, 32) for k in ("z", "u", "v", "t", "q")},
        metadata=Metadata(
            lat=torch.linspace(90, -90, 17),
            lon=torch.linspace(0, 360, 32 + 1)[:-1],
            time=(datetime(2020, 6, 1, 12, 0),),
            atmos_levels=(100, 250, 500, 850),
        ),
    )

    steps = 10

    with torch.inference_mode():
        preds1 = list(rollout(model1, batch, steps))
        preds2 = list(rollout(model2, batch, steps))

    assert len(preds1) == steps
    assert len(preds2) == steps

    for i in range(steps):
        pred1 = preds1[i]
        pred2 = preds2[i]

        expected_time = tuple(t + (i + 1) * timedelta(hours=6) for t in batch.metadata.time)
        assert pred1.metadata.time == expected_time
        assert pred2.metadata.time == expected_time
        assert pred1.metadata.rollout_step == i + 1
        assert pred2.metadata.rollout_step == i + 1

        # The first steps should be equal, but higher steps should not.
        if i == 0:
            assert np.allclose(pred1.surf_vars["2t"], pred2.surf_vars["2t"], rtol=1e-4)
        else:
            assert not np.allclose(pred1.surf_vars["2t"], pred2.surf_vars["2t"], rtol=1e-4)
