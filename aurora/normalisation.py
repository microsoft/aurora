"""Copyright (c) Microsoft Corporation. Licensed under the MIT license."""

from functools import partial
from typing import Optional

import torch

__all__ = [
    "normalise_surf_var",
    "normalise_atmos_var",
    "unnormalise_surf_var",
    "unnormalise_atmos_var",
]


def normalise_surf_var(
    x: torch.Tensor,
    name: str,
    stats: Optional[dict[str, tuple[float, float]]] = None,
    unnormalise: bool = False,
) -> torch.Tensor:
    """Normalise a surface-level variable."""
    if stats and name in stats:
        location, scale = stats[name]
    else:
        location = locations[name]
        scale = scales[name]
    if unnormalise:
        return x * scale + location
    else:
        return (x - location) / scale


def normalise_atmos_var(
    x: torch.Tensor,
    name: str,
    atmos_levels: tuple[int | float, ...],
    unnormalise: bool = False,
) -> torch.Tensor:
    """Normalise an atmospheric variable."""
    level_locations: list[int | float] = []
    level_scales: list[int | float] = []
    for level in atmos_levels:
        level_locations.append(locations[f"{name}_{level}"])
        level_scales.append(scales[f"{name}_{level}"])
    location = torch.tensor(level_locations, dtype=x.dtype, device=x.device)
    scale = torch.tensor(level_scales, dtype=x.dtype, device=x.device)

    if unnormalise:
        return x * scale[..., None, None] + location[..., None, None]
    else:
        return (x - location[..., None, None]) / scale[..., None, None]


unnormalise_surf_var = partial(normalise_surf_var, unnormalise=True)
unnormalise_atmos_var = partial(normalise_atmos_var, unnormalise=True)


locations: dict[str, float] = {
    "z": -1.386496e03,
    "lsm": 0.000000e00,
    "slt": 0.000000e00,
    "2t": 2.785140e02,
    "10u": -5.135059e-02,
    "10v": 1.891580e-01,
    "msl": 1.009578e05,
    "z_50": 1.993730e05,
    "z_100": 1.576421e05,
    "z_150": 1.331414e05,
    "z_200": 1.153300e05,
    "z_250": 1.012231e05,
    "z_300": 8.941415e04,
    "z_400": 6.998038e04,
    "z_500": 5.411537e04,
    "z_600": 4.064833e04,
    "z_700": 2.892882e04,
    "z_850": 1.374978e04,
    "z_925": 7.015005e03,
    "z_1000": 7.381545e02,
    "u_50": 5.653076e00,
    "u_100": 1.027951e01,
    "u_150": 1.354061e01,
    "u_200": 1.420915e01,
    "u_250": 1.334584e01,
    "u_300": 1.180173e01,
    "u_400": 8.817291e00,
    "u_500": 6.563273e00,
    "u_600": 4.814521e00,
    "u_700": 3.345237e00,
    "u_850": 1.418379e00,
    "u_925": 6.172657e-01,
    "u_1000": -3.328723e-02,
    "v_50": 4.226111e-03,
    "v_100": 1.411897e-02,
    "v_150": -3.697671e-02,
    "v_200": -4.507801e-02,
    "v_250": -2.980338e-02,
    "v_300": -2.294770e-02,
    "v_400": -1.771003e-02,
    "v_500": -2.387986e-02,
    "v_600": -2.716674e-02,
    "v_700": 2.153583e-02,
    "v_850": 1.428150e-01,
    "v_925": 2.053480e-01,
    "v_1000": 1.867637e-01,
    "t_50": 2.124864e02,
    "t_100": 2.084042e02,
    "t_150": 2.133201e02,
    "t_200": 2.180615e02,
    "t_250": 2.227710e02,
    "t_300": 2.288696e02,
    "t_400": 2.421368e02,
    "t_500": 2.529492e02,
    "t_600": 2.611347e02,
    "t_700": 2.674010e02,
    "t_850": 2.745600e02,
    "t_925": 2.773572e02,
    "t_1000": 2.810130e02,
    "q_50": 2.678180e-06,
    "q_100": 2.633677e-06,
    "q_150": 5.254625e-06,
    "q_200": 1.940632e-05,
    "q_250": 5.773618e-05,
    "q_300": 1.273861e-04,
    "q_400": 3.855659e-04,
    "q_500": 8.529599e-04,
    "q_600": 1.541429e-03,
    "q_700": 2.431637e-03,
    "q_850": 4.575618e-03,
    "q_925": 6.033134e-03,
    "q_1000": 7.030342e-03,
}

scales: dict[str, float] = {
    "z": 5.884467e04,
    "lsm": 1.000000e00,
    "slt": 7.000000e00,
    "2t": 2.122036e01,
    "10u": 5.547512e00,
    "10v": 4.765339e00,
    "msl": 1.332246e03,
    "z_50": 5.875553e03,
    "z_100": 5.510640e03,
    "z_150": 5.823912e03,
    "z_200": 5.820169e03,
    "z_250": 5.536585e03,
    "z_300": 5.091916e03,
    "z_400": 4.150851e03,
    "z_500": 3.353187e03,
    "z_600": 2.695808e03,
    "z_700": 2.136436e03,
    "z_850": 1.470321e03,
    "z_925": 1.228997e03,
    "z_1000": 1.072307e03,
    "u_50": 1.529281e01,
    "u_100": 1.352611e01,
    "u_150": 1.604335e01,
    "u_200": 1.767630e01,
    "u_250": 1.796710e01,
    "u_300": 1.711917e01,
    "u_400": 1.434276e01,
    "u_500": 1.198419e01,
    "u_600": 1.033421e01,
    "u_700": 9.168821e00,
    "u_850": 8.188043e00,
    "u_925": 7.940808e00,
    "u_1000": 6.141778e00,
    "v_50": 7.058931e00,
    "v_100": 7.479310e00,
    "v_150": 9.571990e00,
    "v_200": 1.188069e01,
    "v_250": 1.338039e01,
    "v_300": 1.334044e01,
    "v_400": 1.122955e01,
    "v_500": 9.181708e00,
    "v_600": 7.803569e00,
    "v_700": 6.871040e00,
    "v_850": 6.264443e00,
    "v_925": 6.470644e00,
    "v_1000": 5.308203e00,
    "t_50": 1.026284e01,
    "t_100": 1.252901e01,
    "t_150": 8.928709e00,
    "t_200": 7.189547e00,
    "t_250": 8.529282e00,
    "t_300": 1.071679e01,
    "t_400": 1.269102e01,
    "t_500": 1.306447e01,
    "t_600": 1.342046e01,
    "t_700": 1.476523e01,
    "t_850": 1.558880e01,
    "t_925": 1.608798e01,
    "t_1000": 1.713983e01,
    "q_50": 3.571687e-07,
    "q_100": 5.703754e-07,
    "q_150": 3.794077e-06,
    "q_200": 2.267534e-05,
    "q_250": 7.446644e-05,
    "q_300": 1.684361e-04,
    "q_400": 5.078644e-04,
    "q_500": 1.079294e-03,
    "q_600": 1.769722e-03,
    "q_700": 2.549169e-03,
    "q_850": 4.112368e-03,
    "q_925": 5.071058e-03,
    "q_1000": 5.913548e-03,
}
