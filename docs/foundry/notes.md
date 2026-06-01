## Resources

* [Documentation of the Aurora model](https://microsoft.github.io/aurora/intro.html)
* [Aurora 1.5 example notebook](https://microsoft.github.io/aurora/example_v1p5.html)
* [Implementation of the Aurora model](https://github.com/microsoft/aurora)
* [Paper with detailed evaluation](https://www.nature.com/articles/s41586-025-09005-y)

## Quickstart

First install the model:

```bash
pip install microsoft-aurora
```

Then you can make predictions with an Azure Foundry AI endpoint as follows:

```python
from aurora import Batch

from aurora.foundry import BlobStorageChannel, FoundryClient, submit


initial_condition = Batch(...)  # Create initial condition for the model.

for pred in submit(
    initial_condition,
    model_name="aurora-0.25-v1.5",
    num_steps=4,  # Every major step predicts six hours ahead.
    fine_lead_times=[1, 2, 3, 4, 5, 6],  # Specify sub-steps like this for hourly forecasts.
    foundry_client=FoundryClient(
        endpoint="https://endpoint/",
        token="ENDPOINT_TOKEN",
    ),
    # Communication with the endpoint happens via an intermediate blob storage container. You
    # will need to create one and generate an URL with a SAS token that has both read and write
    # rights.
    channel=BlobStorageChannel(
        "https://storageaccount.blob.core.windows.net/container?<READ_WRITE_SAS_TOKEN>"
    ),
):
    pass  # Do something with `pred`, which is a `Batch`.
```

## Intended Use

### Primary Use Cases

Aurora 1.5 is intended for medium-range deterministic and probabilistic weather prediction. The
model can be further adapted to specialised atmospheric forecasting tasks with relatively limited
task-specific training data, if desired, due to its foundation-model architecture.

Aurora 1.5 extends the original Aurora with substantially expanded surface variables (26 input,
7 output-only), variable lead-time embeddings enabling predictions at any lead time as fine as
one hour, and an ensemble version for probabilistic forecasting.

The ensemble version (`AuroraV1p5Ensemble`) supports stochastic noise injection for generating
physically plausible ensemble members, producing temporally coherent probabilistic forecasts
suitable for uncertainty quantification in weather prediction research.

### Out-of-Scope Use Cases

Aurora is not designed or evaluated for direct operational decision-making without expert review,
applications requiring guaranteed forecast accuracy, or non-environmental prediction tasks.
Use in safety-critical planning or automated decision pipelines should be accompanied by
appropriate domain validation. Developers should assess the suitability of the model for their
specific downstream use case, and evaluate and mitigate for accuracy and reliability before
deployment, particularly for high-risk scenarios.

## Responsible AI Considerations

Aurora is a research forecasting model and should not be treated as an operational weather service.
While the model can match or exceed traditional numerical baselines on established benchmarks,
its reliability can degrade in out-of-distribution conditions
(e.g., rare extremes, regime shifts, or regions/variables with limited historical fidelity
in underlying reanalyses and simulations).
Outputs may also be misinterpreted if users overlook uncertainty, ensemble spread,
or known limitations of the underlying training data.

The primary Responsible AI risks are
(1) unintended use by non-experts and
(2) downstream use in consequential decision-making without domain validation
(e.g., emergency response, critical infrastructure operations, safety-of-life planning).
To mitigate these risks, developers should:
clearly communicate that the release is for research evaluation and reproducibility;
require domain-expert review before any real-world decisions are informed by outputs;
implement basic input validation to ensure initial conditions come from credible sources
(e.g., established meteorological agencies and data providers);
and benchmark performance against accepted physical modelling systems for the specific
geography, horizon, and variable(s) relevant to the intended application.

Aurora is best integrated as a decision-support component for expert analysis,
not as a fully autonomous trigger for actions.
For higher-risk scenarios, apply additional safeguards such as human-in-the-loop review,
conservative thresholds for alerts, calibration/verification tests,
and ongoing monitoring for drift when changing input sources or pre/post-processing.

## Training Data

The models included in the code have been trained on a variety of publicly available data,
totalling over 1,000,000 hours of heterogeneous Earth-system data spanning atmosphere,
air quality, and ocean-wave domains.
A description of all data, including download links, can be found in
[Supplementary C of the paper](https://www.nature.com/articles/s41586-025-09005-y).
The checkpoints include data from
ERA5, CMIP6 (CMCC-CM2-VHR4 and ECMWF-IFS-HR), HRES forecasts, GFS T0, GFS forecasts,
HRES T0, HRES analysis, HRES-WAM analysis, CAMS reanalysis, and CAMS analysis.

For Aurora 1.5, the model was further fine-tuned via three stages:
(1) adding additional surface variables and increasing temporal resolution to 1 hour,
(2) injecting Gaussian noise during training and optimizing a CRPS objective to improve
probabilistic calibration, and
(3) auto-regressive fine-tuning over multiple 6-hour steps on ECMWF HRES operational analysis (2018–2023).
Unless otherwise noted, the training data are sourced from ERA5 spanning 1981-2023.

## License

This model and the associated model weights are released under the [MIT licence](https://spdx.org/licenses/MIT).

## Security

See [SECURITY](https://github.com/microsoft/aurora/blob/main/SECURITY.md).

## Responsible AI Transparency Documentation

An AI system includes not only the technology, but also the people who will use it,
the people who will be affected by it, and the environment in which it is deployed.
Creating a system that is fit for its intended purpose requires an understanding of how the technology works,
its capabilities and limitations, and how to achieve the best performance.
Microsoft has a broad effort to put our AI principles into practice.

To find out more, see [Responsible AI principles from Microsoft](https://www.microsoft.com/en-us/ai/responsible-ai).

### Limitations

Although Aurora was trained to accurately predict future weather, air pollution, and ocean waves,
Aurora is based on neural networks, which means that there are no strict guarantees that predictions will always be accurate.
Altering the inputs, providing a sample that was not in the training set,
or even providing a sample that was in the training set but is simply unlucky may result in arbitrarily poor predictions.
In addition, even though Aurora was trained on a wide variety of data sets,
it is possible that Aurora inherits biases present in any one of those data sets.
A forecasting system like Aurora is only one piece of the puzzle in a weather prediction pipeline,
and its outputs are not meant to be directly used by people or businesses to plan their operations.
A series of additional verification tests are needed before it can become operationally useful.

## Trademarks

This project may contain trademarks or logos for projects, products, or services.
Authorized use of Microsoft trademarks or logos is subject to and must follow
[Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general).
Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship.
Any use of third-party trademarks or logos are subject to those third-party's policies.
