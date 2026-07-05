# SpML: Spatial Machine Learning

[![Continuous Integration](https://github.com/pysal/spml/actions/workflows/testing.yml/badge.svg)](https://github.com/pysal/spml/actions/workflows/testing.yml)
[![codecov](https://codecov.io/gh/pysal/spml/branch/main/graph/badge.svg)](https://codecov.io/gh/pysal/spml)
[![PyPI version](https://badge.fury.io/py/spml.svg)](https://badge.fury.io/py/spml)
[![Conda Version](https://img.shields.io/conda/vn/conda-forge/spml.svg)](https://anaconda.org/conda-forge/spml)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18173180.svg)](https://doi.org/10.5281/zenodo.18173180)
[![Discord](https://img.shields.io/badge/Discord-join%20chat-7289da?style=flat&logo=discord&logoColor=cccccc&link=https://discord.gg/BxFTEPFFZn)](https://discord.gg/BxFTEPFFZn)
[![SPEC 0 — Minimum Supported Dependencies](https://img.shields.io/badge/SPEC-0-green?labelColor=%23004811&color=%235CA038)](https://scientific-python.org/specs/spec-0000/)

Spatial modeling based on `scikit-learn`.

The aim of the package is to provide implementations of spatially-explicit modelling.

## Features

At the moment, `spml` provides a framework for prototyping geographically weighted extensions of
regression and classification models based on `scikit-learn` and `libpysal.graph` and a
subset of models implemented on top of this framework. For example, you can run
geographically weighted linear regression in a following manner.

```py
import geopandas as gpd
from geodatasets import get_path

from spml.linear_model import GWLinearRegression


gdf = gpd.read_file(get_path('geoda.guerry'))

adaptive = GWLinearRegression(
    bandwidth=25,
    fixed=False,
    kernel='bisquare'
)
adaptive.fit(
    gdf[['Crm_prp', 'Litercy', 'Donatns', 'Lottery']],
    gdf["Suicids"],
    geometry=gdf.representative_point(),
)
```

For details, see the [documentation](https://pysal.org/spml).

## Status

Current development status is beta. The core API of the package should not change
without a warning and a proper deprecation cycle. However, minor breaking changes may
still occur.

## Installation

You can install spatial ML from PyPI or from conda-forge using the tool of your choice:

```sh
pip install spml
```

Or from conda-forge:

```sh
conda install spml -c conda-forge
```

## Bug reports

To search for or report bugs, please see the
[Github issue tracker](https://github.com/pysal/spml/issues).

## Get in touch

If you have a question regarding `spml`, feel free to open an issue or join a chat on
[Discord](https://discord.gg/he6Y8D2ap3).

## License

The package is licensed under BSD 3-Clause License (Copyright (c) 2025, Martin
Fleischmann & PySAL Developers)

## Funding

<img src="https://github.com/pysal/spml/raw/refs/heads/main/docs/source/_static/UK-logo-square-EN.svg" width="200" alt="Charles University logo">

Charles University’s Primus programme through the project "Influence of Socioeconomic and Cultural Factors on Urban Structure in Central Europe", project reference `PRIMUS/24/SCI/023`.
