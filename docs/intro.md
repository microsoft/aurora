# Aurora: A Foundation Model for the Earth System

Welcome to the documentation of Aurora!
Here you will detailed instructions for using the model.
If you just want to see the model in action, you can skip to [a full-fledged example that runs the model on ERA5](example_era5).
For details on how exactly the model works, [please see the paper on arXiv.](https://arxiv.org/abs/2405.13063)

Aurora is a machine learning model that can predict atmospheric variables, such as temperature.
It is a _foundation model_, which means that it was first generally trained on a lot of data,
and then can adapted to specialised atmospheric forecasting tasks with relatively little data.
We provide four such specialised versions:
one for medium-resolution weather prediction,
one for high-resolution weather prediction,
one for air pollution prediction,
and one for ocean wave prediction.

The package currently includes the pretrained model and the fine-tuned version for high-resolution weather forecasting.
We are working on the fine-tuned versions for air pollution and ocean wave forecasting, which will be included in due time.

Cite us as follows:

```
@misc{bodnar2024aurora,
    title = {A Foundation Model for the Earth System},
    author = {Cristian Bodnar and Wessel P. Bruinsma and Ana Lucic and Megan Stanley and Anna Allen and Johannes Brandstetter and Patrick Garvan and Maik Riechert and Jonathan A. Weyn and Haiyu Dong and Jayesh K. Gupta and Kit Thambiratnam and Alexander T. Archibald and Chun-Chieh Wu and Elizabeth Heider and Max Welling and Richard E. Turner and Paris Perdikaris},
    year = {2024},
    url = {https://arxiv.org/abs/2405.13063},
    eprint = {2405.13063},
    archivePrefix = {arXiv},
    primaryClass = {physics.ao-ph},
}
```
