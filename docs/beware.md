# Beware!

When using Aurora, there are a few things to watch out for.

Did you experience an issue that should be listed here?
Please let us know by [opening an issue](https://github.com/microsoft/aurora/issues/new)!

## Sensitivity to Data

Our hope is that Aurora generally produces sensible predictions.
However, there is no guarantee that it will.

For optimal performance, need to use exactly be right.
This means that you should use provide
exactly the right variables
at exactly the right pressure levels
from exactly the right source.
right variables at the right pressure levels

## Deterministic and Reproducible Output

If you require deterministic and reproducible output,
you should do two things:

1. Set `torch.use_deterministic_algorithms(True)` to make PyTorch operations deterministic.

2. Set `model.eval()` to disable drop-out.
