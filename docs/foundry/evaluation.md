All versions of Aurora were extensively evaluated by evaluating predictions on data not seen during training.
These evaluations compare measures of accuracy, such as the root mean square error and anomaly correlation coefficient,
and also examine behaviour in extreme situations, like extreme heat and cold, and rare events, like Storm Ciarán in 2023.
Aurora outperforms operational forecasts across multiple domains, including global air-quality forecasting
(outperforming the baseline in 74% of cases), ocean-wave forecasting (exceeding numerical simulations on 86% of targets),
tropical cyclone track prediction (beating seven operational forecasting centres in 100% of tests),
and high-resolution weather forecasting (surpassing leading models in 92% of scenarios, especially during extreme events).
These evaluations are the main topic of [the paper](https://www.nature.com/articles/s41586-025-09005-y).

For Aurora 1.5, the model was further evaluated on ensemble prediction capabilities.
The Aurora ensemble reduced average CRPS (Continuous Ranked Probability Score) by approximately 3–9%
across key variables compared to operational ensemble baselines.
Case studies (January 2026 winter storm; Hurricane Hélène 2024) illustrate that ensemble diversity
can better capture plausible outcome scenarios for high-impact events.

*Note: The documentation included here is for informational purposes only and is not intended to supersede the applicable license terms.*
