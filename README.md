<img src="docs/aurora.jpg" alt="Aurora logo" width="200"/>

# Aurora: A Foundation Model of the Atmosphere

[![CI](https://github.com/microsoft/Aurora/actions/workflows/ci.yaml/badge.svg)](https://github.com/microsoft/Aurora/actions/workflows/ci.yaml)
[![Latest documentation](https://img.shields.io/badge/docs-latest-blue.svg)](https://microsoft.github.io/aurora)

Implementation of the Aurora model for atmospheric forecasting.

[Link to the paper on arXiv.](https://arxiv.org/abs/2405.13063)

[Link to the documentation.](https://microsoft.github.io/aurora)

Cite us as follows:

```
@misc{bodnar2024aurora,
    title = {Aurora: A Foundation Model of the Atmosphere},
    author = {Cristian Bodnar and Wessel P. Bruinsma and Ana Lucic and Megan Stanley and Johannes Brandstetter and Patrick Garvan and Maik Riechert and Jonathan Weyn and Haiyu Dong and Anna Vaughan and Jayesh K. Gupta and Kit Tambiratnam and Alex Archibald and Elizabeth Heider and Max Welling and Richard E. Turner and Paris Perdikaris},
    year = {2024},
    url = {https://arxiv.org/abs/2405.13063},
    eprint = {2405.13063},
    archivePrefix = {arXiv},
    primaryClass = {physics.ao-ph},
}
```

## Getting Started

Install with `pip`:

```bash
pip install microsoft-aurora
```

Example here.

## FAQ

FAQ.

## Responsible AI Transparency Documentation

An AI system includes not only the technology, but also the people who will use it, the people who will be affected by it, and the environment in which it is deployed.
Creating a system that is fit for its intended purpose requires an understanding of how the technology works, its capabilities and limitations, and how to achieve the best performance.
Microsoft has a broad effort to put our AI principles into practice.
To find out more, see [Responsible AI principles from Microsoft](https://www.microsoft.com/en-us/ai/responsible-ai).

### Use of this code
Our goal in publishing this code is
(1) to facilitate reproducibility of our paper and
(2) to support and accelerate further research into foundation model for atmospheric forecasting.
This code has not been developed nor tested for non-academic purposes and hence should not be used as such.

### No guarantees about quality of predictions
Although Aurora was trained to accurately predict future weather and air pollution,
Aurora is based on neural networks, which means that there are no strict guarantees that predicts will always be accurate.
Altering the inputs to Aurora, providing a sample that was not in the training set,
or even providing a sample that was in the training set but is simply unlucky may result in arbitrarily poor predictions.

### Data
The models included in the code have been trained on a variety of publicly available data.
A description of all data, including download links, can be found in [Supplementary C of the paper](https://arxiv.org/pdf/2405.13063).

*Note: The documentation included in this file is for informational purposes only and is not intended to supersede the applicable license terms.*

## Developing Locally

First, install the repository in editable mode and setup `pre-commit`:

```bash
make install
```

To run the tests and print coverage, run

```bash
make test
```

You can then explore the coverage in the browser by opening `htmlcov/index.html`.

To locally build the documentation, run

```bash
make docs
```

To locally view the documentation, open `docs/_build/index.html` in your browser.
