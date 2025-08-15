from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import geopandas
    import pandas
    from cartopy.mpl.geoaxes import GeoAxes
    from matplotlib.axes import Axes


def sort_ew(cs: geopandas.GeoDataFrame):
    """Sort the frame east to west descending, using the centroid lon value."""
    # TODO: optional reset_index ?
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            category=UserWarning,
            message="Geometry is in a geographic CRS. Results from 'centroid' are likely incorrect.",
        )
        # fmt: off
        return (
            cs
            .assign(x=cs.geometry.centroid.x)
            .sort_values("x", ascending=False)
            .drop(columns="x")
        )
        # fmt: on


def plot_tracked(
    cs: geopandas.GeoDataFrame,
    *,
    alpha: float = 0.25,
    background: str = "countries",
    label: str = "id",
    add_colorbar: bool = False,
    cbar_kwargs: dict | None = None,
    ax: Axes | GeoAxes | None = None,
    size: float = 4,
    aspect: float | None = None,
):
    """Plot CEs at a range of times (colors) with CE group ID (MCS ID) identified.

    Parameters
    ----------
    cs
        Tracked CEs (with MCS ID column).
    alpha
        Alpha applied when plotting the CEs.
    background : {"map", "countries", "none"}
        "map" uses Mercator projection Cartopy's stock image.
        "countries" uses Mercator projection and adds Cartopy coastlines and country borders.
        "none" plots without projection or background.
        This setting is only relevant if `ax` is not provided.
    label : {"id", "none"}
        "id": label each CE with its MCS ID.
        "none": don't label CEs.
    add_colorbar
        Add colorbar with time info.
    cbar_kwargs
        Keyword arguments to pass to ``plt.colorbar``.
    ax
        Axes to plot on. If not provided, a new figure is created.
        The figure size used is ``(size * aspect, size)``.
    size
        Height of the figure (in inches) if `ax` is not provided.
    aspect
        Figure width : height.
        If not provided, it is estimated using the ``total_bounds`` of `cs`.
    """

    import matplotlib.pyplot as plt
    import pandas as pd
    from matplotlib import patheffects

    valid_backgrounds = {"map", "countries", "none"}
    if background not in valid_backgrounds:
        raise ValueError(f"`background` must be one of {valid_backgrounds}")

    valid_labels = {"id", "none"}
    if label not in valid_labels:
        raise ValueError(f"`label` must be one of {valid_labels}")

    if aspect is None:
        # TODO: maybe better to use `.envelope`
        x0, y0, x1, y1 = cs.total_bounds
        aspect = (x1 - x0) / (y1 - y0)
        # ^ estimate for controlling figure size like in xarray
        # https://xarray.pydata.org/en/stable/user-guide/plotting.html#controlling-the-figure-size

    blob_kwargs = dict(alpha=alpha, lw=1.5)
    text_kwargs = dict(
        fontsize=14,
        zorder=10,
        path_effects=[patheffects.withStroke(linewidth=2, foreground="0.2")],
    )
    if ax is None:
        if background in {"map", "countries"}:
            try:
                import cartopy.crs as ccrs
            except ImportError as e:
                raise RuntimeError("cartopy required") from e

            proj = ccrs.Mercator()
            tran = ccrs.PlateCarree()
            fig = plt.figure(figsize=(size * aspect, size))
            ax = fig.add_subplot(projection=proj)
            if TYPE_CHECKING:
                assert isinstance(ax, GeoAxes)
            ax.set_extent([x0, x1, y0, y1])  # type: ignore[attr-defined]
            ax.gridlines(draw_labels=True)  # type: ignore[attr-defined]

            if background == "map":
                # TODO: a more high-res image
                ax.stock_img()  # type: ignore[attr-defined]
            else:  # countries
                import cartopy.feature as cfeature

                ax.add_feature(cfeature.BORDERS, linewidth=0.7, edgecolor="0.3")  # type: ignore[attr-defined]
                ax.coastlines()  # type: ignore[attr-defined]

            blob_kwargs.update(transform=tran)
            text_kwargs.update(transform=tran)

        else:  # none
            _, ax = plt.subplots()

            ax.set(xlabel="lon [°E]", ylabel="lat [°N]")

    t = pd.Series(sorted(cs.time.unique()))
    tmin, tmax = t.iloc[0], t.iloc[-1]
    dt = t.diff().min()

    def get_color(t_):
        if tmin == tmax:
            return plt.cm.tab10.colors[0]
        else:
            x = (t_ - tmin) / (tmax - tmin) * 0.65 + 0.2
            return plt.cm.GnBu(x)

    # Plot blobs at each time
    for t_, g in cs.groupby("time"):
        color = get_color(t_)
        blob_kwargs.update(facecolor=color, edgecolor=color)
        text_kwargs.update(color=color)

        g.plot(ax=ax, **blob_kwargs)

        # Label blobs with assigned ID
        if label == "id":
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    category=UserWarning,
                    message="Geometry is in a geographic CRS. Results from 'centroid' are likely incorrect.",
                )
                for id_, x, y in zip(g.mcs_id, g.centroid.x, g.centroid.y):
                    ax.text(x, y, id_, **text_kwargs)  # type: ignore[arg-type]

    if add_colorbar:
        import matplotlib as mpl

        cbar_kwargs_default = {
            "orientation": "horizontal",
            "label": f"hours since {cs.time.min():%Y-%m-%d %H:%M}",
            "shrink": 0.7,
            "aspect": 40,
        }
        if cbar_kwargs is None:
            cbar_kwargs = {}
        cbar_kwargs = {**cbar_kwargs_default, **cbar_kwargs}

        cmap = mpl.colors.ListedColormap([get_color(t_) for t_ in t])
        hours = ((t - tmin).dt.total_seconds() / 3600).values
        hours = np.append(hours, hours[-1] + dt.total_seconds() / 3600)
        m = mpl.cm.ScalarMappable(cmap=cmap, norm=mpl.colors.BoundaryNorm(hours, cmap.N))
        plt.colorbar(m, ax=ax, **cbar_kwargs)


def _the_unique(s: pandas.Series):
    """Return the one unique value or raise ValueError."""
    u = s.unique()
    if u.size == 1:
        return u[0]
    else:
        raise ValueError(f"the Series has more than one unique value: {u}")
