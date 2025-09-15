"""
Test analysis tools (including those used in :func:`tams.classify`).

- reduce data within elements ("data in contours")
- eccentricity
"""

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from matplotlib.patches import Ellipse
from shapely.geometry import Polygon

import tams


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
    ell = Ellipse((1, 1), w, h, angle=np.rad2deg(np.pi / 4))
    p = Polygon(np.asarray(ell.get_verts()))

    b, a = sorted([w, h])
    eps_expected = np.sqrt(1 - b**2 / a**2)

    eps = tams.calc_ellipse_eccen(p)

    if w == h:  # the model gives ~ 0.06 for the circle
        check = dict(abs=0.07)
    else:
        check = dict(rel=1e-3)

    assert eps == pytest.approx(eps_expected, **check)


def test_data_in_contours_methods_same_result(msg_tb0):
    tb = msg_tb0
    cs235 = tams.core._contours_to_gdf(tams.contours(tb, 235))
    cs219 = tams.core._contours_to_gdf(tams.contours(tb, 219))
    cs235, _ = tams.core._size_filter_contours(cs235, cs219)

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
    cs = tams.identify(ds.tb)[0][0]
    data = ds.pr
    cs_precip = tams.data_in_contours(data, cs, method="sjoin")
    assert cs_precip.mean_pr.sum() > 0
    assert (cs_precip.count_pr > 0).all()


def test_data_in_contours_raises_full_nan(mpas):
    data = mpas.isel(time=0).tb
    cs = tams.identify(mpas.isel(time=1).tb)[0][0]
    assert data.isnull().all()
    with pytest.raises(ValueError, match="all null"):
        tams.data_in_contours(data, cs)


def test_data_in_contours_pass_df(msg_tb0):
    tb = msg_tb0
    data_da = tb
    contours = tams.identify(tb)[0][0]

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
    contours = tams.identify(tb)[0][0]

    df = tams.data_in_contours(data, contours, method=method, agg="mean")
    assert tuple(df.columns) == ("mean_tb", "mean_tb_p100")
    np.isclose(df["mean_tb_p100"].astype(float), df["mean_tb"].astype(float) + 100).all()
