# Fine-Tuning

If you wish to fine-tune Aurora for you specific application,
you should use the pretrained version:

```python
from aurora import Aurora

model = Aurora(use_lora=False)  # Model is not fine-tuned.
model.load_checkpoint("wbruinsma/aurora", "aurora-0.25-pretrained.ckpt")
```

More specific instructions coming soon.
