---
"og:description": TAMS (Tracking Algorithm for Mesoscale Convective Systems) in Python
---

```{module} tams

```

# TAMS

TAMS (**T**racking **A**lgorithm for **M**esoscale Convective **S**ystems) in Python and with more flexibility.

The original TAMS is described in {cite:t}`TAMS1.0`.
{cite:t}`AEW-MCS` applied TAMS to African Easterly Wave research.

A paper describing this version of TAMS {cite:p}`TAMS2.0` is currently under review for publication in _GMD_.

Datasets used in the examples can be retrieved with
{func}`tams.data.download_examples`.

```{toctree}
:caption: Examples
:hidden:

examples/sample-satellite-data.ipynb
examples/tams-run.ipynb
examples/tracking-options.ipynb
examples/sample-mpas-ug-data.ipynb
```

```{toctree}
:caption: Reference
:hidden:

api.rst
differences.md
GitHub <https://github.com/knubez/TAMS>
```

## Installing

TAMS is [available on conda-forge](https://anaconda.org/conda-forge/tams).

```{prompt} bash
conda install -c conda-forge tams
```

[The recipe](https://github.com/conda-forge/tams-feedstock/blob/main/recipe/meta.yaml)
includes the core dependencies and some extras, but you may also wish to install:

- `pyarrow` -- to save results
  with {abbr}`CE (cloud element)` or MCS shapes
  in {class}`~geopandas.GeoDataFrame` format
  to disk as Parquet files with {meth}`~geopandas.GeoDataFrame.to_parquet`
- `ipykernel` -- to use your Conda env in other env's Juptyer

### Development install

If you want to modify the code, you can first clone the repo
and then do an editable install to the dev conda environment:

```{prompt} bash
git clone https://github.com/knubez/TAMS.git
cd TAMS
conda env create -f environment-dev.yml
conda activate tams-dev
pip install -e . --no-deps
```

## References

```{bibliography}
:all:
```
