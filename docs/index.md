# TAMS

TAMS (**T**racking **A**lgorithm for **M**esoscale Convective **S**ystems) in Python and with more flexibility.

The original TAMS is described in {cite:t}`TAMS1.0`.
{cite:t}`AEW-MCS` applied TAMS to African Easterly Wave research.

```{toctree}
:caption: Examples
:hidden:

examples/sample-satellite-data.ipynb
```

```{toctree}
:caption: Reference
:hidden:

api.rst
differences.md
GitHub <https://github.com/knubez/TAMS>
```

## Installing

To install TAMS, first create a
[conda environment](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html)
with at least the core dependencies listed
[here](https://github.com/knubez/TAMS/blob/main/environment-dev.yml).
You may want to include the extras as well.

Then, you can install TAMS directly:

```sh
pip install --no-deps --force-reinstall https://github.com/knubez/TAMS/archive/main.zip
```

Or if you want to modify the code, you can first clone the repo
and then do an editable install:

```sh
git clone https://github.com/knubez/TAMS.git
pip install -e TAMS --no-deps
```

## References

```{bibliography}
:all:
```
