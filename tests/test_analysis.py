"""
Test analysis tools (including those used in :func:`tams.classify`).

- reduce data within elements ("data in contours")
- eccentricity
"""

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely import Polygon

import tams
from tams.core import fit_ellipse
from tams.idealized import Blob


def make_ellipse_polygon(w, h, theta):
    if w == h:
        theta = 0  # no effect, avoid warning

    return Blob.from_wh(w=w, h=h, theta=theta).polygon


@pytest.mark.parametrize(
    "wh",
    [
        (1, 1),
        (1, 0.5),
        (0.5, 1),
        (0.2, 1),
    ],
)
def test_ellipse_eccen(wh):
    w, h = wh
    p = make_ellipse_polygon(w, h, theta=45)

    b2, a2 = sorted([w, h])
    eps_expected = np.sqrt(1 - b2**2 / a2**2)

    eps = tams.eccentricity(p)

    with pytest.warns(FutureWarning, match="`calc_ellipse_eccen` has been renamed"):
        eps_deprecated = tams.calc_ellipse_eccen(p)

    assert eps == eps_deprecated, "same func"

    if w == h:
        check = dict(abs=1e-7)
    else:
        check = dict(rel=1e-12)

    assert eps == pytest.approx(eps_expected, **check)


def test_ellipse_eccen_invalid():
    with (
        pytest.warns(
            RuntimeWarning,  # from scikit-image
            match=r"Need at least 5 data points to estimate an ellipse\.",
        ),
        pytest.warns(
            UserWarning,
            match="ellipse fitting failed for",
        ),
    ):
        res = tams.eccentricity(Polygon([]))
    assert np.isnan(res)

    with (
        pytest.warns(
            RuntimeWarning,  # from scikit-image
            match=r"Standard deviation of data is too small to estimate ellipse with meaningful precision\.",
        ),
        pytest.warns(
            UserWarning,
            match="ellipse fitting failed for",
        ),
    ):
        res = tams.eccentricity(Polygon([(0, 0)] * 5))
    assert np.isnan(res)

    # scikit-image message (not surfaced in v0.26): "Singular matrix from estimation"
    with pytest.warns(
        UserWarning,
        match="ellipse fitting failed for",
    ):
        res = tams.eccentricity(Polygon([(i, i) for i in range(10)]))
    assert np.isnan(res)


def test_ellipse_fit_blob():
    b = Blob(a=3, b=1, theta=10)
    m = fit_ellipse(b.polygon)
    assert m is not None
    for k in ["c", "a", "b", "theta"]:
        v0 = getattr(b, k)
        v = getattr(m, k)
        assert v == pytest.approx(v0, rel=1e-12), k


def test_data_in_contours_methods_same_result(msg_tb0):
    tb = msg_tb0
    cs235_ = tams.core._contours_to_polygons(tams.contour(tb, 235))
    cs219 = tams.core._contours_to_polygons(tams.contour(tb, 219))
    cs235 = tams.core._size_filter(cs235_, cs219)
    assert 0 < len(cs235) < len(cs235_)

    vn = "tb"
    varnames = [vn]
    x1 = tams.core._data_in_contours_sjoin(tb, cs235, varnames=varnames)
    x2 = tams.core._data_in_contours_regionmask(tb, cs235, varnames=varnames)
    assert len(x1) == len(x2)
    assert (x1[f"count_{vn}"] > 0).all()
    assert x1[f"count_{vn}"].equals(x2[f"count_{vn}"])
    # Note: With pandas v1, std's were all the same exactly, but not mean
    # With pandas v2, the opposite
    assert x1[f"mean_{vn}"].equals(x2[f"mean_{vn}"])
    dstd = x1[f"std_{vn}"] - x2[f"std_{vn}"]
    assert len(x1) - dstd.eq(0).sum() in {3, 5}
    assert dstd.abs().max() < 2e-6


def test_data_in_contours_non_xy(mpas):
    # The MPAS one has real lat/lon 1-D dim-coords, not x/y with 2-D lat/lon
    # so with `to_dataframe().reset_index(drop=True)` lat/lon were lost
    ds = mpas.isel(time=1)
    cs = tams.identify(ds.tb)[0]
    data = ds.pr
    cs_precip = tams.data_in_contours(data, cs, method="sjoin")
    assert cs_precip.mean_pr.sum() > 0
    assert (cs_precip.count_pr > 0).all()


def test_data_in_contours_raises_full_nan(mpas):
    data = mpas.isel(time=0).tb
    cs = tams.identify(mpas.isel(time=1).tb)[0]
    assert data.isnull().all()
    with pytest.raises(ValueError, match="all null"):
        tams.data_in_contours(data, cs)


def test_data_in_contours_pass_df(msg_tb0):
    tb = msg_tb0
    data_da = tb
    contours = tams.identify(tb)[0]

    data_ds = data_da.to_dataset()
    data_df = data_da.to_dataframe().reset_index(drop=True)  # drop (lat, lon) index
    data_gdf = gpd.GeoDataFrame(
        data_df,
        geometry=gpd.points_from_xy(data_df.lon, data_df.lat),
        crs="EPSG:4326",
    )

    in_contours_data_da = tams.data_in_contours(data_da, contours)
    in_contours_data_ds = tams.data_in_contours(data_ds, contours)
    in_contours_data_df = tams.data_in_contours(data_df, contours)
    in_contours_data_gdf = tams.data_in_contours(data_gdf, contours)

    results = [
        in_contours_data_da,
        in_contours_data_ds,
        in_contours_data_df,
        in_contours_data_gdf,
    ]
    for res in results:
        assert isinstance(res, pd.DataFrame), "just df with merge=False"
    for left, right in zip(results[:-1], results[1:]):
        assert left is not right
        pd.testing.assert_frame_equal(left, right)


@pytest.mark.parametrize("method", ["sjoin", "regionmask"])
def test_data_in_contours_pass_ds_multiple_vars(msg_tb0, method):
    tb = msg_tb0
    # TODO: rename to 'tb' in `tb_from_ir` (and update examples/tests)
    data = tb.rename("tb").to_dataset().assign(tb_p100=tb + 100)
    contours = tams.identify(tb)[0]

    df = tams.data_in_contours(data, contours, method=method, agg="mean")
    assert tuple(df.columns) == ("mean_tb", "mean_tb_p100")
    np.isclose(df["mean_tb_p100"].astype(float), df["mean_tb"].astype(float) + 100).all()
