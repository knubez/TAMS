from __future__ import annotations

from collections.abc import Generator
from datetime import datetime

import pybtex.plugin
from pybtex.database import Entry, Person
from pybtex.style.labels import BaseLabelStyle
from pybtex.style.sorting import BaseSortingStyle

project = "tams"
html_title = "TAMS"
html_logo = "_static/TAMS-logo.png"
author = "K. M. Núñez Ocasio and Z. Moon"
copyright = "2022\u20132026, K. M. Núñez Ocasio and Z. Moon"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.extlinks",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
    "sphinxcontrib.bibtex",
    "myst_nb",
    "sphinxext.opengraph",
    "sphinx_prompt",
    "sphinx_copybutton",
    "sphinx_togglebutton",
]

nitpicky = True

exclude_patterns = [
    "_build/**/*",
    "**.ipynb_checkpoints",
    # "api/**/*",
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
    "cartopy": ("https://cartopy.readthedocs.io/stable/", None),
}

extlinks = {
    "issue": ("https://github.com/knubez/TAMS/issues/%s", "GH%s"),
    "pull": ("https://github.com/knubez/TAMS/pull/%s", "PR%s"),
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
    "Ellipse": "~matplotlib.patches.Ellipse",
    "Triangulation": "~matplotlib.tri.Triangulation",
    "Callable": "~typing.Callable",
    "GeoSeries": "geopandas.GeoSeries",
    "GeoDataFrame": "geopandas.GeoDataFrame",
    "gpd.GeoSeries": "geopandas.GeoSeries",
    "gpd.GeoDataFrame": "geopandas.GeoDataFrame",
    "GeoAxes": "~cartopy.mpl.geoaxes.GeoAxes",
    "Path": "~pathlib.Path",
    "datetime-like": ":func:`datetime-like <pandas.to_datetime>`",
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
    "duck-array": ":term:`duck-array <duck array>`",
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


class SortingStyle(BaseSortingStyle):
    """Sort by date descending, then by author name, then title.
    Each entry in our bib should have `year` to set a `%b %Y` format date
    (e.g. 'Jan 2023').
    """

    # https://bitbucket.org/pybtex-devs/pybtex/src/9b97822f5517fc7893456b9827589a003ea7076a/pybtex/style/sorting/author_year_title.py?at=master

    def persons_key(self, persons: list[Person]) -> str:
        return "   ".join(self.person_key(person) for person in persons)

    def person_key(self, person: Person) -> str:
        return "  ".join(
            (
                " ".join(person.prelast_names + person.last_names),
                " ".join(person.first_names + person.middle_names),
                " ".join(person.lineage_names),
            )
        ).lower()

    def sorting_key(self, entry: Entry) -> tuple:
        author_key = self.persons_key(entry.persons["author"])
        year_entry = entry.fields["year"]
        date = datetime.strptime(year_entry, r"%b %Y").date()
        title_key = entry.fields["title"].replace("{", "")
        key = (date, author_key, title_key)
        return key

    def sort(self, entries: list[Entry]) -> list[Entry]:
        return sorted(entries, key=self.sorting_key, reverse=True)


class LabelStyle(BaseLabelStyle):
    """Use the sorting as given but label in reverse."""

    # https://github.com/mcmtroffaes/sphinxcontrib-bibtex/blob/a455b0e07f81d6448356a446443a97101486778e/test/roots/test-bibliography_style_label_1/conf.py
    # https://bitbucket.org/pybtex-devs/pybtex/src/9b97822f5517fc7893456b9827589a003ea7076a/pybtex/style/labels/number.py?at=master#lines-33

    def format_labels(self, sorted_entries: list[Entry]) -> Generator[str]:
        n = len(sorted_entries)
        for i, _ in enumerate(sorted_entries):
            yield str(n - i)


pybtex.plugin.register_plugin("pybtex.style.sorting", bibtex_default_style, SortingStyle)
pybtex.plugin.register_plugin("pybtex.style.labels", bibtex_default_style, LabelStyle)

# Override class defaults
# Since sphinxcontrib-bibtex doesn't (seem to) give us a way to set
# `sorting_style` and `label_style`, etc. for the formatting class init
# https://bitbucket.org/pybtex-devs/pybtex/src/9b97822f5517fc7893456b9827589a003ea7076a/pybtex/style/formatting/__init__.py?at=master#lines-47
from pybtex.style.formatting.plain import Style

# Original: 'author_year_title'
Style.default_sorting_style = bibtex_default_style

# Original: None
Style.default_label_style = bibtex_default_style


def setup(app):
    def skip_member(app, what, name, obj, skip, options):
        if name.startswith("_"):
            return True

        # print(f"{name=}, {what=}, {obj=}")

        # Skip inherited (named) tuple methods
        if what == "method":
            qualname = getattr(obj, "__qualname__", "")

            if qualname.startswith("tuple."):
                return True

        return skip

    app.connect("autodoc-skip-member", skip_member)
