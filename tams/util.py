from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import geopandas
    import matplotlib
    import pandas


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
    ax: matplotlib.axes.Axes | None = None,
    size: float = 4,
):
    """Plot CEs at a range of times (colors) with CE group ID (MCS ID) identified."""

    import matplotlib.pyplot as plt
    from matplotlib import patheffects

    valid_backgrounds = {"map", "countries", "none"}
    if background not in valid_backgrounds:
        raise ValueError(f"`background` must be one of {valid_backgrounds}")

    valid_labels = {"id", "none"}
    if label not in valid_labels:
        raise ValueError(f"`label` must be one of {valid_labels}")

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
            ax.set_extent([x0, x1, y0, y1])
            ax.gridlines(draw_labels=True)

            if background == "map":
                # TODO: a more high-res image
                ax.stock_img()
            else:  # countries
                import cartopy.feature as cfeature

                ax.add_feature(cfeature.BORDERS, linewidth=0.7, edgecolor="0.3")
                ax.coastlines()

            blob_kwargs.update(transform=tran)
            text_kwargs.update(transform=tran)

        else:  # none
            _, ax = plt.subplots()

            ax.set(xlabel="lon [°E]", ylabel="lat [°N]")

    nt = cs.time.unique().size
    colors = plt.cm.GnBu(np.linspace(0.2, 0.85, nt))

    # Plot blobs at each time
    for i, (_, g) in enumerate(cs.groupby("time")):
        color = colors[i]
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
                    ax.text(x, y, id_, **text_kwargs)


def _the_unique(s: pandas.Series):
    """Return the one unique value or raise ValueError."""
    u = s.unique()
    if u.size == 1:
        return u[0]
    else:
        raise ValueError(f"the Series has more than one unique value: {u}")
