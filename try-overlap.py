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
        # TODO: original TAMS normalized by the *min* area between a and b, could offer option
        res[i] = ov.to_dict()

    return res


r = tams.load_example_ir()

tb = tams.tb_from_ir(r, ch=9)

nt = tb.time.size
assert nt > 1
itimes = list(range(nt))

# Compute assumed duration of each time step based on the dataset time resolution
dti = pd.DatetimeIndex(tb.time)
dt = dti[1:] - dti[:-1]
assert (dt.astype(int) > 0).all()
if not dt.unique().size == 1:
    warnings.warn("unequal time spacing")
dt = dt.insert(-1, dt[-1])

# tb0 = tb.isel(time=0)
# tb1 = tb.isel(time=1)

# cs0 = tams.identify(tb0)
# cs1 = tams.identify(tb1)

# # For each in cs0, check overlap with all in cs1
# res = overlap(cs0, cs1)

# IDEA: even at initial time, could put CEs together in groups based on edge-to-edge distance

# Loop over available times
css = []
for l in itimes[:4]:  # noqa: E741
    tb_l = tb.isel(time=l)
    cs_l = tams.identify(tb_l)
    cs_l["time"] = tb_l.time.values
    cs_l["itime"] = l
    cs_l["duration"] = dt[l]
    n_l = len(cs_l)
    if l == 0:  # noqa: E741
        # IDs all new for first time step
        cs_l["id"] = range(n_l)
        i_id = len(cs_l)  # next ID

    else:
        # Assign IDs using overlap threshold
        thresh = 0.5
        u = -5  # 5--13 m/s are typical values to use
        cs_lm1 = css[l - 1]

        assert "time" in cs_lm1.columns and "time" in cs_l.columns
        try:
            (t_lm1,) = cs_lm1.time.unique()
            (t_l,) = cs_l.time.unique()
        except ValueError as e:
            raise ValueError("expected single times") from e
        dt_l = pd.Timedelta(t_l - t_lm1).total_seconds()
        assert dt_l > 0

        # TODO: option to overlap in other direction, match with all that meet the condition
        ovs = overlap(cs_l, project(cs_lm1, u=u, dt=dt_l))
        ids = []
        for i, d in ovs.items():
            # TODO: option to pick bigger one to "continue the trajectory", as in jevans paper
            j, frac = max(d.items(), key=lambda tup: tup[1], default=(None, 0))
            if j is None or frac < thresh:
                # New ID
                ids.append(i_id)
                i_id += 1
            else:
                # Has "parent"; use their "family" ID
                ids.append(cs_lm1.loc[j].id)

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


# %% Classify


def calc_eccen(p):
    """Compute the eccentricity of the best-fit ellipse to the Polygon's exterior.

    Parameters
    ----------
    p : shapely.geometry.Polygon
    """
    # Based on https://scipython.com/book/chapter-8-scipy/examples/non-linear-fitting-to-an-ellipse/
    from scipy import optimize

    x, y = np.asarray(p.exterior.coords).T
    r = np.hypot(x, y)
    theta = np.arctan2(y, x)

    def f(p):
        # a - semi-major axis length
        # e - eccentricity
        a, e = p
        return a * (1 - e**2) / (1 - e * np.cos(theta))

    def residuals(p):
        return r - f(p)

    def jac(p):  # of the residuals
        a, e = p
        da = (1 - e**2) / (1 - e * np.cos(theta))
        de = (-2 * a * e * (1 - e * np.cos(theta)) + a * (1 - e**2) * np.cos(theta)) / (
            1 - e * np.cos(theta)
        ) ** 2
        # return -da,  -de
        return np.column_stack((-da, -de))

    # a0 = np.mean(r)
    a0 = np.ptp(r) * 0.7
    e0 = 0.2  # note that a circle has e of 0 and 1 is the max
    # plsq = optimize.leastsq(residuals, x0=(a0, e0), Dfun=jac, col_deriv=True)
    # return plsq[0]#[1]

    # TODO: try with newer `least_squares` interface
    # print(a0, e0)
    # print(theta)
    # print(np.ptp(r), np.ptp(theta))
    res = optimize.least_squares(
        residuals,
        x0=(a0, e0),
        bounds=((0, 0), (r.max() * np.ptp(theta), 1)),
        jac=jac,
    )
    # return res.x
    # r_ = f(res.x)
    a, e = res.x
    theta_ = np.linspace(0, 2 * np.pi, 100)
    r_ = a * (1 - e**2) / (1 - e * np.cos(theta_))
    return r_ * np.cos(theta_), r_ * np.sin(theta_)


def calc_eccen2(p):
    """Compute the eccentricity of the best-fit ellipse to the Polygon's exterior.

    Parameters
    ----------
    p : shapely.geometry.Polygon
    """
    # using skimage https://scikit-image.org/docs/dev/api/skimage.measure.html#skimage.measure.EllipseModel
    from skimage.measure import EllipseModel

    xy = np.asarray(p.exterior.coords)
    assert xy.shape[1] == 2

    m = EllipseModel()
    m.estimate(xy)
    _, _, xhw, yhw, _ = m.params
    # ^ xc, yc, a, b, theta; from the docs
    #   a with x, b with y (after subtracting the rotation), but they are half-widths
    #   theta is in radians

    rat = yhw / xhw if xhw > yhw else xhw / yhw

    return np.sqrt(1 - rat**2)


from matplotlib.patches import Ellipse
from shapely.geometry import Polygon
from skimage.measure import EllipseModel

fig, ax = plt.subplots()

for w, h, c in [(1, 1, "r"), (0.5, 1, "g"), (1, 0.5, "b")]:
    ell = Ellipse((1, 1), w, h, np.rad2deg(np.pi / 4), color=c, alpha=0.3)
    p = Polygon(ell.get_verts())
    b, a = sorted([w, h])
    eps = np.sqrt(1 - b**2 / a**2)
    print(
        eps,
        eps / np.sqrt(1 - eps**2),
        eps / np.sqrt(2 - eps**2),
        np.arcsin(eps),
        "|",
        calc_eccen2(p),
    )

    ax.add_patch(ell)
    # ax.plot(*calc_eccen(p), "o-", color=c)

    m = EllipseModel()
    m.estimate(np.asarray(p.exterior.coords))
    xc, yc, a, b, theta = m.params
    ell2 = Ellipse((xc, yc), 2 * a, 2 * b, np.rad2deg(theta), ec=c, fc="none", ls=":", lw=2)
    ax.add_patch(ell2)

ax.set(xlim=(0, 2), ylim=(0, 2))
ax.axis("equal")


# eps = sqrt(1 - (b^2/a^2)) -- ellipse "first eccentricity"
#
# Below from most to least strict:
#
# MCCs (organized)
# - 219 K region >= 25k km2
# - 235 K region >= 50k km2
# - size durations have to be met for >= 6 hours
# - eps <= 0.7
#
# CCCs (organized)
# - 219 K region >= 25k km2
# - size durations have to be met for >= 6 hours
# - no shape criterion
#
# DLL (disorganized)
# - >= 6 hour duration
# - (no size or shape criterion)
#
# DSL (disorganized)
# - < 6 hour duration
#
# Classification is for the "family" groups


def _the_unique(s: pd.Series):
    """Return the one unique value or raise ValueError."""
    u = s.unique()
    if u.size == 1:
        return u[0]
    else:
        raise ValueError(f"the Series has more than one unique value: {u}")


def classify(cs: gpd.GeoDataFrame) -> str:
    assert cs.id.unique().size == 1, "for a certain family group"

    # Sum areas over cloud elements
    time_groups = cs.groupby("time")
    area = time_groups[["area_km2", "area219_km2"]].apply(sum)

    # # Assume duration from the time index
    # dt = area.index[1:] - area.index[:-1]
    # if not dt.unique().size == 1:
    #     warnings.warn("unequally spaced times")
    # dt = dt.insert(-1, dt[-1])

    # Get duration
    dt = time_groups["duration"].apply(_the_unique)

    # Compute area-duration criteria
    dur_219_25k = dt[area.area219_km2 >= 25_000].sum()
    dur_235_50k = dt[area.area_km2 >= 50_000].sum()
    six_hours = pd.Timedelta(hours=6)

    if dur_219_25k >= six_hours:  # organized
        # Compute ellipse eccentricity
        eps = time_groups[["geometry"]].apply(
            lambda g: calc_eccen2(g.dissolve().geometry.convex_hull.iloc[0])
        )
        dur_eps = dt[eps <= 0.7].sum()
        if dur_235_50k >= six_hours and dur_eps >= six_hours:
            class_ = "MCC"
        else:
            class_ = "CCC"

    else:  # disorganized
        if dt.sum() >= six_hours:
            class_ = "DLL"
        else:
            class_ = "DSL"

    return class_


classes = cs.groupby("id").apply(classify)
cs["class"] = cs.id.map(classes)

plt.show()
