# Fine-Tuning

Generally, if you wish to fine-tune Aurora for a specific application,
you should build on the pretrained version:

```python
from aurora import Aurora

model = Aurora(use_lora=False)  # Model is not fine-tuned.
model.load_checkpoint("microsoft/aurora", "aurora-0.25-pretrained.ckpt")
```

## Extending Aurora with New Variables

Aurora can be extended with new variables by adjusting the keyword arguments `surf_vars`,
`static_vars`, and `atmos_vars`.
When you add a new variable, you also need to set the normalisation statistics.

```python
from aurora import Aurora
from aurora.normalisation import locations, scales

model = Aurora(
    use_lora=False,
    surf_vars=("2t", "10u", "10v", "msl", "new_surf_var"),
    static_vars=("lsm", "z", "slt", "new_static_var"),
    atmos_vars=("z", "u", "v", "t", "q", "new_atmos_var"),
)
model.load_checkpoint("microsoft/aurora", "aurora-0.25-pretrained.ckpt")

# Normalisation means:
locations["new_surf_var"] = 0.0
locations["new_static_var"] = 0.0
locations["new_atmos_var"] = 0.0

# Normalisation standard deviations:
scales["new_surf_var"] = 1.0
scales["new_static_var"] = 1.0
scales["new_atmos_var"] = 1.0
```

## Other Model Extensions

It is possible to extend to model in any way you like.
If you do this, you will likely add or remove parameters.
Then `Aurora.load_checkpoint` will error,
because the existing checkpoint now mismatches with the model's parameters.
Simply set `Aurora.load_checkpoint(..., strict=False)` to ignore the mismatches:

```python
from aurora import Aurora

model = Aurora(...)

... # Modify `model`.

model.load_checkpoint("microsoft/aurora", "aurora-0.25-pretrained.ckpt", strict=False)
```
