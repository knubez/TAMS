---
"og:description": TAMS (Tracking Algorithm for Mesoscale Convective Systems) in Python
---

```{module} tams

```

# TAMS

TAMS (**T**racking **A**lgorithm for **M**esoscale Convective **S**ystems) in Python and with more flexibility.

The original TAMS is described in {cite:t}`TAMS1.0`.
{cite:t}`AEW-MCS` applied TAMS to African Easterly Wave research.

Datasets used in the examples can be retrieved with
{func}`tams.data.download_examples`.

```{toctree}
:caption: Examples
:hidden:

examples/sample-satellite-data.ipynb
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
