# Aurora: A Foundation Model for the Earth System

Welcome to the documentation of Aurora!
Here you will detailed instructions for using the model.
If you just want to see the model in action, you can skip to [a full-fledged example that runs the model on ERA5](example_era5).
For details on how exactly the model works, [please see the paper](https://www.nature.com/articles/s41586-025-09005-y).

Aurora is a machine learning model that can predict atmospheric variables, such as temperature.
It is a _foundation model_, which means that it was first generally trained on a lot of data,
and then can adapted to specialised atmospheric forecasting tasks with relatively little data.
We provide four such specialised versions:
one for medium-resolution weather prediction,
one for high-resolution weather prediction,
one for air pollution prediction,
and one for ocean wave prediction.

Cite us as follows:

```
@article{bodnar2025aurora,
    title = {A Foundation Model for the Earth System},
    author = {Cristian Bodnar and Wessel P. Bruinsma and Ana Lucic and Megan Stanley and Anna Allen and Johannes Brandstetter and Patrick Garvan and Maik Riechert and Jonathan A. Weyn and Haiyu Dong and Jayesh K. Gupta and Kit Thambiratnam and Alexander T. Archibald and Chun-Chieh Wu and Elizabeth Heider and Max Welling and Richard E. Turner and Paris Perdikaris},
    journal = {Nature},
    year = {2025},
    month = {May},
    day = {21},
    issn = {1476-4687},
    doi = {10.1038/s41586-025-09005-y},
    url = {https://doi.org/10.1038/s41586-025-09005-y},
}
```

Please email [AIWeatherClimate@microsoft.com](mailto:AIWeatherClimate@microsoft.com)
if you are interested in using Aurora for commercial applications.
For research-related questions or technical support with the code here,
please [open an issue](https://github.com/microsoft/aurora/issues/new/choose)
or reach out to the authors of the paper.
