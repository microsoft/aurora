# Tropical Cyclone Tracking

Aurora has the ability to track tropical cyclones (TCs).
For tracking TCs, we recommend to use Aurora 0.25° Fine-Tuned.
The tracker is available as `aurora.Tracker`.
It should be used in conjunction with `aurora.rollout`.

Here is an example:

```python
from datetime import datetime

from aurora import Aurora, Batch, Tracker, rollout

model = Aurora()
model.load_checkpoint()

# Construct an initial condition for the model. The TC will be tracked using
# predictions for this initial condition.
initial_condition = Batch(...)

# Initialise the tracker with the current position and time of the TC. The time
# should match with the above initial condition.
tracker = Tracker(init_lat=..., init_lon=..., init_time=datetime(...))

model.eval()
model = model.to("cuda")

# Run the tracker for predictions up to two days (8 six-hour steps).
with torch.inference_mode():
    for pred in rollout(model, batch, steps=8):
        tracker.step(pred)

model = model.to("cpu")
```

Afterwards, the track can be conveniently summarised in a DataFrame:

```python
track = tracker.results()
```

[Here is a full example](example_tc_tracking) that runs the tracker to track
[Typhoon Nanmadol](https://en.wikipedia.org/wiki/Typhoon_Nanmadol_(2022)).
