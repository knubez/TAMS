import sys

sys.path.append("../")


project = "tams"
copyright = "2022, K. M. Núñez Ocasio and TAMS developers"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.extlinks",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
    "myst_nb",
]

html_theme = "sphinx_book_theme"

intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "xarray": ("https://xarray.pydata.org/en/stable/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
    "geopandas": ("https://geopandas.org/en/stable/", None),
    "dask": ("https://docs.dask.org/en/latest", None),
}

napoleon_numpy_docstring = True
napoleon_preprocess_types = True
napoleon_use_param = False
napoleon_use_rtype = False

napoleon_type_aliases = {
    "DataArray": "~xarray.DataArray",
    "Dataset": "~xarray.Dataset",
    "Figure": "~matplotlib.figure.Figure",
    "Axes": "~matplotlib.axes.Axes",
    "Callable": "~typing.Callable",
}

nb_execution_mode = "cache"
nb_execution_excludepatterns = [
    "t2.ipynb",
    "t3.ipynb",
    "_build/**/*",
]
exclude_patterns = nb_execution_excludepatterns
