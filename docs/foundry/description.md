Aurora is a machine learning foundation model designed to predict atmospheric and environmental variables
such as temperature and wind speed.
It is pretrained on diverse weather and climate data and subsequently fine-tuned for specialised
environmental forecasting tasks including weather prediction, air pollution modelling, and ocean
wave forecasting (see [our Nature paper](https://www.nature.com/articles/s41586-025-09005-y)).
Aurora 1.5 extends the original Aurora architecture with 22 new single-level output variables
(including radiation fluxes, precipitation, and 100-m winds) and variable lead-time embeddings
enabling predictions at any lead time as fine as one hour. An ensemble version is also provided and
adds stochastic noise injection for generating physically plausible ensemble members for
probabilistic forecasting. For more details, please see the
[Aurora documentation](https://microsoft.github.io/aurora/intro.html)
and [Aurora 1.5 example](https://microsoft.github.io/aurora/example_v1p5.html).

Please email [AIWeatherClimate@microsoft.com](mailto:AIWeatherClimate@microsoft.com)
if you are interested in using Aurora for commercial applications.
For research-related questions or technical support with the open-source version of the model,
please [open an issue in the GitHub repository](https://github.com/microsoft/aurora/issues/new/choose)
or reach out to the authors of the paper.
