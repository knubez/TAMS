from __future__ import annotations

import logging
import os
import sys
import warnings
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import geopandas
    import pandas
    from cartopy.mpl.geoaxes import GeoAxes
    from matplotlib.axes import Axes
    from matplotlib.colors import Colormap


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


def cmap_section(
    cmap: str | Colormap,
    start: float = 0.0,
    stop: float = 1.0,
    num: int = 256,
) -> Colormap:
    """Sample a portion of a colormap to make a new one."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    if not 0 <= start < stop <= 1:
        raise ValueError("invalid start/stop values. Must be in [0, 1].")

    cmap_ = plt.get_cmap(cmap)

    return ListedColormap(
        cmap_(np.linspace(start, stop, num)),
        name=f"{cmap}_{start}_{stop}",
    )


def plot_tracked(
    cs: geopandas.GeoDataFrame,
    *,
    alpha: float = 0.25,
    background: str = "countries",
    label: str = "id",
    add_colorbar: bool = False,
    cmap: str | Colormap | None = None,
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
        Only allowed if there is more than one unique time.
    cmap : str or Colormap, optional
        Colormap, used to indicate the time of each CE
        relative to the min and max time in the frame.
        The default is the 0.2--0.85 section of ``'GnBu'``,
        such that the earliest CEs are light green.
        Note that the upper bound will be used if there is only one unique time.
    cbar_kwargs
        Keyword arguments to pass to ``plt.colorbar``.
    ax : Axes or GeoAxes, optional
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
            fig = plt.figure(figsize=(size * aspect, size))
            ax = fig.add_subplot(projection=proj)
            if TYPE_CHECKING:
                assert isinstance(ax, GeoAxes)
            ax.set_extent([x0, x1, y0, y1])
            ax.gridlines(draw_labels=True)

            if background == "map":
                # TODO: a more high-res image
                ax.stock_img()
            else:  # countries
                import cartopy.feature as cfeature

                ax.add_feature(cfeature.BORDERS, linewidth=0.7, edgecolor="0.3")
                ax.coastlines()

        else:  # none
            _, ax = plt.subplots()

            ax.set(xlabel="lon [°E]", ylabel="lat [°N]")

    try:
        import cartopy.crs as ccrs
        from cartopy.mpl.geoaxes import GeoAxes
    except ImportError:
        pass
    else:
        if isinstance(ax, GeoAxes):
            tran = ccrs.PlateCarree()

            blob_kwargs.update(transform=tran)
            text_kwargs.update(transform=tran)

    t = pd.Series(sorted(cs.time.unique()))
    tmin, tmax = t.iloc[0], t.iloc[-1]
    dt = t.diff().min()
    if tmin == tmax and add_colorbar:
        raise ValueError("adding colorbar when there is only one unique time is not supported")

    if cmap is None:
        cmap = cmap_section("GnBu", 0.2, 0.85)

    cmap_obj = plt.get_cmap(cmap)

    def get_color(t_):
        if tmin == tmax:
            return cmap_obj(1.0)
        else:
            x = (t_ - tmin) / (tmax - tmin)
            return cmap_obj(x)

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


def get_logger() -> logging.Logger:
    """Get the ``'tams'`` logger."""
    logger = logging.getLogger("tams")

    return logger


def set_logger_level(level: int | str) -> None:
    """Set the logging level for the "tams" logger.

    Parameters
    ----------
    level
        Logging level, e.g., ``0`` (unset), ``logging.DEBUG``, ``'INFO'``, ``logging.WARNING``.
    """
    logger = get_logger()
    logger.setLevel(level)


def set_logger_handler(
    *,
    stderr: bool = False,
    stdout: bool = False,
    file: str | Path | None = None,
) -> None:
    """Set logging handler(s) for the "tams" logger.
    By default, resets to no handlers.

    Examples
    --------
    To stderr:
    >>> set_logger_handler(stderr=True)

    To file:
    >>> set_logger_handler(file="tams.log")

    Reset to nothing:
    >>> set_logger_handler()
    """
    logger = get_logger()

    if stderr and stdout:
        raise ValueError("only one of `stderr` and `stdout` can be True")

    for h in logger.handlers:
        logger.removeHandler(h)

    fmt_file = "%(levelname)s:%(asctime)s - %(message)s"
    fmt_console = f"%(name)s:{fmt_file}"

    handler: logging.Handler
    if stderr:
        handler = logging.StreamHandler(sys.stderr)
        formatter = logging.Formatter(fmt_console)
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    elif stdout:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(fmt_console)
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    if file is not None:
        handler = logging.FileHandler(file)
        formatter = logging.Formatter(fmt_file)
        handler.setFormatter(formatter)
        logger.addHandler(handler)


def get_worker_logger():
    """Get logger for a worker process, adapting to the joblib backend.

    Special cases:

    * serial execution: the ``'tams'`` logger will be returned, unmodified
      - :func:`get_logger`
    * Dask: the ``'distributed.worker'`` logger will be returned, unmodified
      - ``client.get_worker_logs()``
    * Ray: the ``'ray'`` logger will be returned, unmodified
      - https://docs.ray.io/en/latest/ray-observability/user-guides/configure-logging.html

    Otherwise (joblib loky, multiprocessing, threading backends),
    the ``'tams.worker'`` logger will be returned,
    configured to indicate process and/or thread ID in the log messages,
    and handler/level consistent with the TAMS options.

    Configure Dask or Ray logging beforehand.
    This can be as simple as::

        logging.getLogger('distributed.worker').setLevel(logging.DEBUG)

    or::

        logging.getLogger('ray').setLevel(logging.DEBUG)
    """
    from multiprocessing import current_process
    from threading import current_thread

    # For Dask or Ray, just return their logger,
    # rely on their native logging.
    try:
        from dask.distributed import get_client

        _ = get_client()
        return logging.getLogger("distributed.worker")
    except (ImportError, ValueError):
        pass
    try:
        import ray

        if ray.is_initialized():
            return logging.getLogger("ray")
    except (ImportError, ValueError):
        pass

    # Standard joblib backends or serial execution
    fmt_file = "%(levelname)s:%(asctime)s - %(message)s"
    this_process = current_process()
    this_thread = current_thread()
    if this_process.name != "MainProcess":
        # Joblib process-based (loky or multiprocessing)
        fmt_file = "%(processName)s:" + fmt_file
    elif this_thread.name != "MainThread":
        # Threading
        fmt_file = "%(threadName)s:" + fmt_file
    else:
        # Serial
        return get_logger()

    # For joblib, use worker logger
    logger = logging.getLogger("tams.worker")
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
    logger.propagate = False

    # Set handler
    handler_setting = os.getenv("TAMS_WORKER_LOGGER_HANDLER", "").strip()
    if handler_setting != "":
        if handler_setting == "stderr":
            h = logging.StreamHandler(sys.stderr)
            fmt = f"%(name)s:{fmt_file}"
        elif handler_setting == "stdout":
            h = logging.StreamHandler(sys.stdout)
            fmt = f"%(name)s:{fmt_file}"
        else:
            h = logging.FileHandler(handler_setting)
            fmt = fmt_file
        h.setFormatter(logging.Formatter(fmt))
        logger.addHandler(h)

    # Set log level
    level_setting = os.getenv("TAMS_WORKER_LOGGER_LEVEL", "").strip()
    try:
        level_setting = int(level_setting)
    except ValueError:
        pass
    if level_setting == "":
        logger.setLevel(logging.NOTSET)
    else:
        logger.setLevel(level_setting)

    # Log worker info
    this_pid = os.getpid()
    main_pid = os.getppid()
    logger.debug(f"{this_pid=}, {main_pid=}, {this_process.name=}, {this_thread.name=}")

    return logger
