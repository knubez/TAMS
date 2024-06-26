---
"og:description": TAMS (Tracking Algorithm for Mesoscale Convective Systems) in Python
---

```{module} tams

```

# TAMS

TAMS (**T**racking **A**lgorithm for **M**esoscale Convective **S**ystems) in Python and with more flexibility.

The original TAMS is described in {cite:t}`TAMS1.0`.
{cite:t}`AEW-MCS` applied TAMS to African Easterly Wave research.

```{note}
A paper describing _this_ implementation of TAMS {cite:p}`TAMS2.0` in Python has been accepted for publication in _GMD_.
```

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
- `ipykernel` -- to use your Conda env in another env's Jupyter

```{attention}
Current {func}`tams.identify` [doesn't work](https://github.com/knubez/TAMS/issues/13)
with `matplotlib` 3.8.0 (mid Sep 2023), but 3.8.1 (end of Oct 2023)
restored the previous behavior.
```

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
