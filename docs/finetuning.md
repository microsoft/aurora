# Fine-Tuning

If you wish to fine-tune Aurora for you specific application,
you should use the pretrained version:

```python
from aurora import Aurora

model = Aurora(use_lora=False)  # Model is not fine-tuned.
model.load_checkpoint("microsoft/aurora", "aurora-0.25-pretrained.ckpt")
```

You are also free to extend the model for your particular use case.
In that case, it might be that you add or remove parameters.
Then `Aurora.load_checkpoint` will error,
because the existing checkpoint now mismatches with the model's parameters.
Simply set `Aurora.load_checkpoint(..., strict=False)`:

```python
from aurora import Aurora


model = Aurora(...)

... # Modify `model`.

model.load_checkpoint("microsoft/aurora", "aurora-0.25-pretrained.ckpt", strict=False)
```

More instructions coming soon!
