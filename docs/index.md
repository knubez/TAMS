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
A paper describing _this_ implementation of TAMS {cite:p}`TAMS2.0` in Python has been published in _GMD_ (2024-08-15).
```

```{toctree}
:caption: Examples
:hidden:

examples/sample-satellite-data.ipynb
examples/tams-run.ipynb
examples/tracking-options.ipynb
examples/sample-mpas-ug-data.ipynb
examples/identify.ipynb
examples/get.ipynb
```

```{toctree}
:caption: Reference
:hidden:

api.rst
changes.md
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
- `ipykernel` -- to use your conda environment in another environment's Jupyter
  (for example, [NCAR JupyterHub](https://jupyterhub.hpc.ucar.edu/))

```{attention}
Current {func}`tams.identify` [doesn't work](https://github.com/knubez/TAMS/issues/13)
with `matplotlib` 3.8.0 (mid Sep 2023), but 3.8.1 (end of Oct 2023)
restored the previous behavior.
```

```{note}
In the past (before TAMS v0.1.5, mid Aug 2024),
the TAMS conda-forge recipe included PyGEOS,
in order to make certain GeoPandas and regionmask operations faster.
[In Shapely v2](https://shapely.readthedocs.io/en/stable/release/2.x.html)
(mid Dec 2022, but not relevant to TAMS
[until mid 2023](https://geopandas.org/en/stable/docs/user_guide/pygeos_to_shapely.html)),
[PyGEOS is part of Shapely](https://shapely.readthedocs.io/en/stable/migration_pygeos.html)
and doesn't need to be installed separately.
GeoPandas dropped support for Shapely v1 and PyGEOS in
[their v1 release](https://github.com/geopandas/geopandas/releases/tag/v1.0.0) (late Jun 2024).
[PyGEOS on conda-forge](https://github.com/conda-forge/pygeos-feedstock) has been retired,
so you likely won't be able to install it in new conda environments in any case.
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

## Citing

If you use TAMS in your research, we would appreciate it if you cite {cite:t}`TAMS2.0`.
Since v0.1.2 (late Sep 2023),
you can additionally cite the specific version of TAMS that you used
[via Zenodo](https://doi.org/10.5281/zenodo.8393890).

## References

```{bibliography}
:filter: key % "TAMS[0-9]"
```

## Papers using TAMS

```{bibliography}
:filter: not key % "TAMS[0-9]"
:labelprefix: U
```

## Example notebook timings

```{nb-exec-table}

```
