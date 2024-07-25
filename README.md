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
