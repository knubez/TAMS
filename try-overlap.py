import warnings

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import tams

plt.close("all")


def sort_ew(cs: gpd.GeoDataFrame):
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


def _project_geometry(s: gpd.GeoSeries, *, dx: float) -> gpd.GeoSeries:
    crs0 = s.crs.to_string()

    return s.to_crs(crs="EPSG:32663").translate(xoff=dx).to_crs(crs0)


# TODO: test


def project(df: gpd.GeoDataFrame, *, u: float = 0, dt: float = 3600):
    """Project the coordinates by `u`*`dt` meters.

    Parameters
    ----------
    u
        Speed [m s-1]
    dt
        Time [s]. Default: one hour.
    """
    dx = u * dt
    new_geometry = _project_geometry(df.geometry, dx=dx)

    return df.assign(geometry=new_geometry)


def overlap(a: gpd.GeoDataFrame, b: gpd.GeoDataFrame):
    """For each contour in `a`, determine those in `b` that overlap and by how much.

    Currently the mapping is based on indices of the frames.
    """
    a_area = a.to_crs("EPSG:32663").area
    res = {}
    for i in range(len(a)):
        a_i = a.iloc[i : i + 1]  # slicing preserves GeoDataFrame type
        a_i_poly = a_i.values[0][0]
        with warnings.catch_warnings():
            # We get this warning when an empty intersection is found
            warnings.filterwarnings(
                "ignore",
                category=RuntimeWarning,
                message="invalid value encountered in intersection",
            )
            inter = b.intersection(a_i_poly)  # .dropna()
        inter = inter[~inter.is_empty]
        ov = inter.to_crs("EPSG:32663").area / a_area.iloc[i]
        res[i] = ov.to_dict()

    return res


r = tams.load_example_ir()

tb = tams.tb_from_ir(r, ch=9)

nt = tb.time.size
itimes = list(range(nt))

# tb0 = tb.isel(time=0)
# tb1 = tb.isel(time=1)

# cs0 = tams.identify(tb0)
# cs1 = tams.identify(tb1)

# # For each in cs0, check overlap with all in cs1
# res = overlap(cs0, cs1)


# Loop over available times
css = []
for l in itimes[:3]:  # noqa: E741
    tb_l = tb.isel(time=l)
    cs_l = tams.identify(tb_l)
    cs_l["time"] = tb_l.time.values
    cs_l["itime"] = l
    n_l = len(cs_l)
    if l == 0:  # noqa: E741
        # IDs all new for first time step
        cs_l["id"] = range(n_l)
        i_id = len(cs_l)  # next ID

    else:
        # Assign IDs using overlap threshold
        # TODO: optional projection velocity
        thresh = 0.5
        ovs = overlap(cs_l, css[l - 1])
        ids = []
        for i, d in ovs.items():
            j, frac = max(d.items(), key=lambda tup: tup[1], default=(None, 0))
            if j is None or frac < thresh:
                # New ID
                ids.append(i_id)
                i_id += 1
            else:
                # Has "parent"; use their "family" ID
                ids.append(css[l - 1].loc[j].id)

        cs_l["id"] = ids

    css.append(cs_l)

# Combine into one frame
cs = pd.concat(css)


# %% Plot

fig, ax = plt.subplots(figsize=(16, 5))

colors = plt.cm.GnBu(np.linspace(0.2, 0.85, nt))

# Plot blobs at each time
for i, g in cs.groupby("itime"):
    color = colors[i]

    # g.plot(ax=ax, fc=color, ec=color, alpha=0.3, lw=1.5)  # color arg aliases doesn't work for gpd
    g.plot(ax=ax, facecolor=color, edgecolor=color, alpha=0.25, lw=1.5)

    # Label blobs with assigned ID
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            category=UserWarning,
            message="Geometry is in a geographic CRS. Results from 'centroid' are likely incorrect.",
        )
        for id_, x, y in zip(g.id, g.centroid.x, g.centroid.y):
            ax.text(x, y, id_, c=color, fontsize=14, zorder=10)


fig.tight_layout()

plt.show()
