# Installation

You can install spatial ML from PyPI or from conda-forge using the tool of your choice:

```sh
pip install spml
```

Or from conda-forge:

```sh
conda install spml -c conda-forge
```

## Installing development version

You can either clone the repository:

```sh
git clone https://github.com/pysal/spml.git
cd spml
pip install .
```

Or install directly from Github:

```sh
pip install git+https://github.com/pysal/spml.git
```

The package depends on:

```yaml
geopandas>=1.1.0
joblib>=1.5.0
libpysal>=4.12
numpy>=2.0.0
scipy>=1.14.0
scikit-learn>=1.5.0
pandas>=2.3.0
```