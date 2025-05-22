"""Copyright (c) Microsoft Corporation. Licensed under the MIT license.

The cumbersome checkpoint wrangling in this file is for making the published checkpoints compatible
with the published versions of the model. You can safely ignore all of this.
"""

import torch

from aurora.normalisation import level_to_str

__all__ = [
    "_adapt_checkpoint_pretrained",
    "_adapt_checkpoint_air_pollution",
]


def _adapt_checkpoint_pretrained(
    patch_size: int,
    d: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    # Remove possibly prefix from the keys.
    for k, v in list(d.items()):
        if k.startswith("net."):
            del d[k]
            d[k[4:]] = v

    # Convert the ID-based parametrization to a name-based parametrization.
    if "encoder.surf_token_embeds.weight" in d:
        weight = d["encoder.surf_token_embeds.weight"]
        del d["encoder.surf_token_embeds.weight"]

        assert weight.shape[1] == 4 + 3
        for i, name in enumerate(("2t", "10u", "10v", "msl", "lsm", "z", "slt")):
            d[f"encoder.surf_token_embeds.weights.{name}"] = weight[:, [i]]

    if "encoder.atmos_token_embeds.weight" in d:
        weight = d["encoder.atmos_token_embeds.weight"]
        del d["encoder.atmos_token_embeds.weight"]

        assert weight.shape[1] == 5
        for i, name in enumerate(("z", "u", "v", "t", "q")):
            d[f"encoder.atmos_token_embeds.weights.{name}"] = weight[:, [i]]

    if "decoder.surf_head.weight" in d:
        weight = d["decoder.surf_head.weight"]
        bias = d["decoder.surf_head.bias"]
        del d["decoder.surf_head.weight"]
        del d["decoder.surf_head.bias"]

        assert weight.shape[0] == 4 * patch_size**2
        assert bias.shape[0] == 4 * patch_size**2
        weight = weight.reshape(patch_size**2, 4, -1)
        bias = bias.reshape(patch_size**2, 4)

        for i, name in enumerate(("2t", "10u", "10v", "msl")):
            d[f"decoder.surf_heads.{name}.weight"] = weight[:, i]
            d[f"decoder.surf_heads.{name}.bias"] = bias[:, i]

    if "decoder.atmos_head.weight" in d:
        weight = d["decoder.atmos_head.weight"]
        bias = d["decoder.atmos_head.bias"]
        del d["decoder.atmos_head.weight"]
        del d["decoder.atmos_head.bias"]

        assert weight.shape[0] == 5 * patch_size**2
        assert bias.shape[0] == 5 * patch_size**2
        weight = weight.reshape(patch_size**2, 5, -1)
        bias = bias.reshape(patch_size**2, 5)

        for i, name in enumerate(("z", "u", "v", "t", "q")):
            d[f"decoder.atmos_heads.{name}.weight"] = weight[:, i]
            d[f"decoder.atmos_heads.{name}.bias"] = bias[:, i]

    return d


def _adapt_checkpoint_air_pollution(
    patch_size: int,
    d: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    if "encoder.surf_token_embeds.weight_new" in d:
        weight = d["encoder.surf_token_embeds.weight_new"]
        del d["encoder.surf_token_embeds.weight_new"]

        assert weight.shape[1] == (3 + 5) + 4 * 2 + 3 * 2
        for i, name in enumerate(
            ("pm1", "pm2p5", "pm10", "tcco", "tc_no", "tcno2", "gtco3", "tcso2")
            + ("static_ammonia", "static_ammonia_log", "static_co", "static_co_log")
            + ("static_nox", "static_nox_log", "static_so2", "static_so2_log")
            + ("tod_cos", "tod_sin", "dow_cos")
            + ("dow_sin", "doy_cos", "doy_sin")
        ):
            d[f"encoder.surf_token_embeds.weights.{name}"] = weight[:, [i]]

    # Now fix the patch embeddings for the atmospheric variables. These are more complicated.

    if (
        "encoder.atmos_token_embeds.weights.z" in d
        and "encoder.atmos_token_embeds_new.layers.50.weight" in d
    ):
        bias = d["encoder.atmos_token_embeds.bias"]
        del d["encoder.atmos_token_embeds.bias"]

        for name in ("z", "u", "v", "t", "q"):
            weight = d[f"encoder.atmos_token_embeds.weights.{name}"]
            del d[f"encoder.atmos_token_embeds.weights.{name}"]

            for level in (50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000):
                # Clone them here to prevent mutation from doing something weird!
                d[f"encoder.atmos_token_embeds.layers.{level}.weights.{name}"] = weight.clone()
                d[f"encoder.atmos_token_embeds.layers.{level}.bias"] = bias.clone()

    n1 = "encoder.atmos_token_embeds.weight_new2"
    if n1 in d:
        weight = d[n1]
        del d[n1]
        for level in (50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000):
            n2 = f"encoder.atmos_token_embeds.layers.{level_to_str(level)}.weights.{{}}"

            assert weight.shape[1] == 17
            # These are all taken from `atmos_token_embeds`!
            for i, name in enumerate(
                ("static_lsm", "static_z", "static_slt")
                # For the atmospheric variables, there is _another_ prefix `static_`.
                + ("static_static_ammonia", "static_static_ammonia_log")
                + ("static_static_co", "static_static_co_log")
                + ("static_static_nox", "static_static_nox_log")
                + ("static_static_so2", "static_static_so2_log")
                + ("static_tod_cos", "static_tod_sin", "static_dow_cos")
                + ("static_dow_sin", "static_doy_cos", "static_doy_sin")
            ):
                d[n2.format(name)] = weight[:, [i]]

    if "encoder.atmos_token_embeds.weight_new" in d:
        del d["encoder.atmos_token_embeds.weight_new"]

    if "encoder.atmos_token_embeds.weight_new2" in d:
        del d["encoder.atmos_token_embeds.weight_new2"]

    for level in (50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000):
        # The patch embedding is doubly specified. Select the right one.
        n1 = f"encoder.atmos_token_embeds_new.layers.{level_to_str(level)}.weight"
        if n1 in d:
            del d[n1]

        n1 = f"encoder.atmos_token_embeds_new.layers.{level_to_str(level)}.weight_new"
        n2 = f"encoder.atmos_token_embeds.layers.{level_to_str(level)}.weights.{{}}"
        if n1 in d:
            weight = d[n1]
            del d[n1]
            assert weight.shape[1] == 5
            for i, name in enumerate(("co", "no", "no2", "go3", "so2")):
                d[n2.format(name)] = weight[:, [i]]

        # This simulates an indexing bug where `z` also uses the patch embedding for `static_z`.
        d[f"encoder.atmos_token_embeds.layers.{level_to_str(level)}.weights.z"] = d[
            f"encoder.atmos_token_embeds.layers.{level_to_str(level)}.weights.static_z"
        ]

        n1 = f"encoder.atmos_token_embeds_new.layers.{level_to_str(level)}.bias"
        n2 = f"encoder.atmos_token_embeds.layers.{level_to_str(level)}.bias"
        if n1 in d:
            assert n2 in d  # The bias is already defined!
            # Because the original implementation had two separate patch embedding instances, we
            # need to add the biases.
            d[n2] += d[n1]
            del d[n1]

        if f"encoder.atmos_token_embeds_new.layers.{level_to_str(level)}.weight_new2" in d:
            del d[f"encoder.atmos_token_embeds_new.layers.{level_to_str(level)}.weight_new2"]

    # Remove the feature combiners for the non-positive variables.
    for name in ("2t", "10u", "10v", "msl"):
        if f"surf_feature_combiner.{name}.weight" in d:
            del d[f"surf_feature_combiner.{name}.weight"]
            del d[f"surf_feature_combiner.{name}.bias"]
        pass
    for name in ("z", "u", "v", "t", "q"):
        if f"atmos_feature_combiner.{name}.weight" in d:
            del d[f"atmos_feature_combiner.{name}.weight"]
            del d[f"atmos_feature_combiner.{name}.bias"]
        pass

    # Rename the second Perceiver in the decoder.
    for k in list(d):
        p1 = "decoder.level_decoder_new"
        p2 = "decoder.level_decoder_alternate"
        if k.startswith(p1):
            d[p2 + k.removeprefix(p1)] = d[k]
            del d[k]

    # Do the same thing that we did for the encoder now for the decoder.

    if "decoder.surf_head_new.weight" in d:
        weight = d["decoder.surf_head_new.weight"]
        bias = d["decoder.surf_head_new.bias"]
        del d["decoder.surf_head_new.weight"]
        del d["decoder.surf_head_new.bias"]

        n = 8
        assert weight.shape[0] == n * patch_size**2
        assert bias.shape[0] == n * patch_size**2
        weight = weight.reshape(patch_size**2, n, -1)
        bias = bias.reshape(patch_size**2, n)

        for i, name in enumerate(
            ("pm1", "pm2p5", "pm10", "tcco", "tc_no", "tcno2", "gtco3", "tcso2")
        ):
            d[f"decoder.surf_heads.{name}.weight"] = weight[:, i]
            d[f"decoder.surf_heads.{name}.bias"] = bias[:, i]

    if "decoder.surf_head_mod.weight" in d:
        weight = d["decoder.surf_head_mod.weight"]
        bias = d["decoder.surf_head_mod.bias"]
        del d["decoder.surf_head_mod.weight"]
        del d["decoder.surf_head_mod.bias"]

        n = 4 + 8
        assert weight.shape[0] == n * patch_size**2
        assert bias.shape[0] == n * patch_size**2
        weight = weight.reshape(patch_size**2, n, -1)
        bias = bias.reshape(patch_size**2, n)

        for i, name in enumerate(
            ("2t", "10u", "10v", "msl")
            + ("pm1", "pm2p5", "pm10", "tcco", "tc_no", "tcno2", "gtco3", "tcso2"),
        ):
            d[f"decoder.surf_heads.{name}_mod.weight"] = weight[:, i]
            d[f"decoder.surf_heads.{name}_mod.bias"] = bias[:, i]

    for suffix in ("", "_mod"):
        for level in (50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000):
            if f"decoder.atmos_head{suffix}.layers.{level}.weight" in d:
                weight = d[f"decoder.atmos_head{suffix}.layers.{level}.weight"]
                bias = d[f"decoder.atmos_head{suffix}.layers.{level}.bias"]
                del d[f"decoder.atmos_head{suffix}.layers.{level}.weight"]
                del d[f"decoder.atmos_head{suffix}.layers.{level}.bias"]

                n = 5
                assert weight.shape[0] == n * patch_size**2
                assert bias.shape[0] == n * patch_size**2
                weight = weight.reshape(patch_size**2, n, -1)
                bias = bias.reshape(patch_size**2, n)

                for i, v in enumerate(("z", "u", "v", "t", "q")):
                    d[f"decoder.atmos_heads.{v}{suffix}.layers.{level}.weight"] = weight[:, i]
                    d[f"decoder.atmos_heads.{v}{suffix}.layers.{level}.bias"] = bias[:, i]

            if f"decoder.atmos_head{suffix}_new.layers.{level}.weight" in d:
                weight = d[f"decoder.atmos_head{suffix}_new.layers.{level}.weight"]
                bias = d[f"decoder.atmos_head{suffix}_new.layers.{level}.bias"]
                del d[f"decoder.atmos_head{suffix}_new.layers.{level}.weight"]
                del d[f"decoder.atmos_head{suffix}_new.layers.{level}.bias"]

                n = 5
                assert weight.shape[0] == n * patch_size**2
                assert bias.shape[0] == n * patch_size**2
                weight = weight.reshape(patch_size**2, n, -1)
                bias = bias.reshape(patch_size**2, n)

                for i, v in enumerate(("co", "no", "no2", "go3", "so2")):
                    d[f"decoder.atmos_heads.{v}{suffix}.layers.{level}.weight"] = weight[:, i]
                    d[f"decoder.atmos_heads.{v}{suffix}.layers.{level}.bias"] = bias[:, i]

    return d
