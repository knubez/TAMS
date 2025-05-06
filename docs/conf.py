import sys

sys.path.append("../")


project = "tams"
html_title = "TAMS"
html_logo = "_static/TAMS-logo.png"
author = "K. M. Núñez Ocasio and Z. Moon"
copyright = "2022\u20132025, K. M. Núñez Ocasio and Z. Moon"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.extlinks",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
    "sphinxcontrib.bibtex",
    "myst_nb",
    "sphinxext.opengraph",
    "sphinx-prompt",
    "sphinx_copybutton",
    "sphinx_togglebutton",
]

exclude_patterns = [
    "_build/**/*",
    "**.ipynb_checkpoints",
    # "api/**/*",
    "examples/t2.ipynb",
]

html_theme = "sphinx_book_theme"

intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "xarray": ("https://docs.xarray.dev/en/stable/", None),
    "pandas": ("https://pandas.pydata.org/pandas-docs/stable/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
    "geopandas": ("https://geopandas.org/en/stable/", None),
    "dask": ("https://docs.dask.org/en/latest/", None),
    "shapely": ("https://shapely.readthedocs.io/en/stable/", None),
    "earthaccess": ("https://earthaccess.readthedocs.io/en/stable/", None),
}

napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_preprocess_types = True
napoleon_use_param = True
napoleon_use_rtype = False

napoleon_type_aliases = {
    "DataArray": "~xarray.DataArray",
    "Dataset": "~xarray.Dataset",
    "Figure": "~matplotlib.figure.Figure",
    "Axes": "~matplotlib.axes.Axes",
    "Callable": "~typing.Callable",
    "gpd.GeoDataFrame": "geopandas.GeoDataFrame",
    # General terms
    "sequence": ":term:`sequence`",
    "iterable": ":term:`iterable`",
    "callable": ":py:func:`callable`",
    "dict_like": ":term:`dict-like <mapping>`",
    "dict-like": ":term:`dict-like <mapping>`",
    "path-like": ":term:`path-like <path-like object>`",
    "mapping": ":term:`mapping`",
    "file-like": ":term:`file-like <file-like object>`",
    # NumPy terms
    "array_like": ":term:`array_like`",
    "array-like": ":term:`array-like <array_like>`",
    "scalar": ":term:`scalar`",
    "array": ":term:`array`",
    "hashable": ":term:`hashable <name>`",
}

nb_execution_mode = "cache"
nb_execution_excludepatterns = exclude_patterns + [
    "examples/tracking-options.ipynb",
]
nb_execution_raise_on_error = True
nb_execution_timeout = 90

myst_enable_extensions = [
    "dollarmath",
    "smartquotes",
    "replacements",
]

autodoc_typehints = "description"
autosummary_generate = True

bibtex_bibfiles = ["refs.bib"]
bibtex_default_style = "plain"

ogp_image = "_static/TAMS-logo.png"
