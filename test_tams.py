from pathlib import Path

import earthaccess
import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import xarray as xr

import tams

r = tams.data.load_example_ir().isel(time=0)

tb = tams.data.tb_from_ir(r, ch=9)

glade_avail = Path("/glade").is_dir()

try:
    # Raising exceptions on login failure is new in v0.14.0 (2025-02-11)
    # https://github.com/nsidc/earthaccess/releases/tag/v0.14.0
    # https://github.com/nsidc/earthaccess/pull/946/files#diff-58622dff7ea01bdbe5d5e0b10dc8d252da4aecccb4e122d3adeb268e227bf155
    from earthaccess.exceptions import LoginAttemptFailure, LoginStrategyUnavailable
except ImportError:
    earthdata_login_failure_allowed = True
else:
    earthdata_login_failure_allowed = False

if earthdata_login_failure_allowed:
    auth = earthaccess.login()
    earthdata = auth.authenticated
else:
    try:
        auth = earthaccess.login(strategy="netrc")
    except (LoginStrategyUnavailable, LoginAttemptFailure):  # no netrc file or invalid
        earthdata = False
    else:
        earthdata = True
skipif_no_earthdata = pytest.mark.skipif(not earthdata, reason="need Earthdata auth")


def test_ch9_tb_loaded():
    assert tb.name == "ch9"
    assert tuple(tb.coords) == ("lon", "lat", "time")


def test_data_in_contours_methods_same_result():
    cs235 = tams.core._contours_to_gdf(tams.contours(tb, 235))
    cs219 = tams.core._contours_to_gdf(tams.contours(tb, 219))
    cs235, _ = tams.core._size_filter_contours(cs235, cs219)

    varnames = ["ch9"]
    x1 = tams.core._data_in_contours_sjoin(tb, cs235, varnames=varnames)
    x2 = tams.core._data_in_contours_regionmask(tb, cs235, varnames=varnames)
    assert len(x1) == len(x2)
    assert (x1.count_ch9 > 0).all()
    assert x1.count_ch9.equals(x2.count_ch9)
    # Note: With pandas v1, std's were all the same exactly, but not mean
    # With pandas v2, the opposite
    assert x1.mean_ch9.equals(x2.mean_ch9)
    dstd = x1.std_ch9 - x2.std_ch9
    assert len(x1) - dstd.eq(0).sum() in {3, 5}
    assert dstd.abs().max() < 2e-6


def test_data_in_contours_non_xy():
    # The MPAS one has real lat/lon 1-D dim-coords, not x/y with 2-D lat/lon
    # so with `to_dataframe().reset_index(drop=True)` lat/lon were lost
    ds = tams.load_example_mpas().isel(time=1)
    cs = tams.identify(ds.tb)[0][0]
    data = ds.precip
    cs_precip = tams.data_in_contours(data, cs, method="sjoin")
    assert cs_precip.mean_precip.sum() > 0
    assert (cs_precip.count_precip > 0).all()


def test_data_in_contours_raises_full_nan():
    data = tams.load_example_mpas().isel(time=0).tb
    cs = tams.identify(tams.load_example_mpas().isel(time=1).tb)[0][0]
    assert data.isnull().all()
    with pytest.raises(ValueError, match="all null"):
        tams.data_in_contours(data, cs)


def test_data_in_contours_pass_df():
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
def test_data_in_contours_pass_ds_multiple_vars(method):
    # TODO: rename to 'tb' in `tb_from_ir` (and update examples/tests)
    data = tb.rename("tb").to_dataset().assign(tb_p100=tb + 100)
    contours = tams.identify(tb)[0][0]

    df = tams.data_in_contours(data, contours, method=method, agg="mean")
    assert tuple(df.columns) == ("mean_tb", "mean_tb_p100")
    np.isclose(df["mean_tb_p100"].astype(float), df["mean_tb"].astype(float) + 100).all()


def test_load_mpas_sample():
    ds = tams.load_example_mpas()
    assert tuple(ds.data_vars) == ("tb", "precip")
    assert tuple(ds.coords) == ("time", "lon", "lat")


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
    from matplotlib.patches import Ellipse
    from shapely.geometry import Polygon

    w, h = wh
    ell = Ellipse((1, 1), w, h, angle=np.rad2deg(np.pi / 4))
    p = Polygon(ell.get_verts())

    b, a = sorted([w, h])
    eps_expected = np.sqrt(1 - b**2 / a**2)

    eps = tams.calc_ellipse_eccen(p)

    if w == h:  # the model gives ~ 0.06 for the circle
        check = dict(abs=0.07)
    else:
        check = dict(rel=1e-3)

    assert eps == pytest.approx(eps_expected, **check)


def test_contour_too_small_skipped():
    # With a few of the sample MPAS data time steps (e.g. `.isel(time=22)`)
    # the current contouring algo returns some with less than 3 points,
    # which can't make a LinearRing (it raises ValueError)

    contours = [np.array([[0, 0], [1, 1]])]
    gdf = tams.core._contours_to_gdf(contours)
    assert len(contours) == 1
    assert len(gdf) == 0

    contours = [np.array([[0, 0], [1, 1], [0, 0.5]])]
    gdf = tams.core._contours_to_gdf(contours)
    assert len(contours) == 1
    assert len(gdf) == 1


# TODO: test to check crs of all geometry columns returned by `run` are correct
# and all have a RangeIndex


@pytest.mark.skipif(not glade_avail, reason="need to have GLADE fs available")
def test_mpas_precip_loader():
    ds = tams.load_mpas_precip(
        "/glade/scratch/rberrios/cpex-aw/"
        "2021082500/intrp_output/mpas_init_2021082500_valid_2021-08-25_*_latlon_wpac.nc"
    )

    assert set(ds.data_vars) == {"tb", "precip"}
    assert tuple(ds.dims) == ("time", "lat", "lon")
    assert ds.sizes["time"] == 24


def test_classify_empty():
    cs = gpd.GeoDataFrame(
        columns=["mcs_id", "geometry", "time", "dtime", "area_km2", "area219_km2"],
        crs="EPSG:4326",
    )
    with pytest.warns(UserWarning, match="empty input frame"):
        cs_ = tams.classify(cs)
        assert "mcs_class" in cs_ and "mcs_class" not in cs


def test_classify_cols_check():
    cs = gpd.GeoDataFrame(
        columns=["mcs_id", "geometry", "time", "area_km2", "area219_km2"],
        data=[[0, None, pd.NaT, np.nan, np.nan]],
        crs="EPSG:4326",
    )
    with pytest.raises(ValueError, match="missing these columns"):
        _ = tams.classify(cs)


def test_identify_no_ces_warning():
    tb_p100 = tb + 100
    with pytest.warns(UserWarning, match="No CEs identified"):
        _ = tams.identify(tb_p100)

    ctt = xr.concat(
        [
            tb,
            tb_p100.assign_coords(time=tb_p100.time + np.timedelta64(1, "h")),
            tb_p100.assign_coords(time=tb_p100.time + np.timedelta64(2, "h")),
        ],
        dim="time",
    )
    with pytest.warns(UserWarning, match=r"No CEs identified for time steps: \[1, 2\]"):
        _ = tams.identify(ctt)


@skipif_no_earthdata
@pytest.mark.parametrize(
    "version,run",
    [
        # ("06", "early"),
        # ("06", "late"),
        ("07", "early"),
        ("07", "late"),
        ("07", "final"),
    ],
)
def test_get_imerg(version, run):
    ds = tams.data.get_imerg("2019-06-01", version=version, run=run)
    assert set(ds.data_vars) == {"pr", "pr_err", "pr_qi"}
    assert set(ds.coords) == {"time", "lat", "lon"}
    for vn in ds.data_vars:
        assert tuple(ds[vn].dims) == ("lat", "lon"), "squeezed"
    assert ds["pr"].isnull().sum() > 0


def test_tams_run_basic():
    # Keep this like the notebook example but can be shorter
    ds = tams.load_example_mpas().rename({"tb": "ctt", "precip": "pr"}).isel(time=slice(1, 7))

    ce, mcs, mcs_summary = tams.run(ds)
    assert isinstance(ce, gpd.GeoDataFrame)
    assert isinstance(mcs, gpd.GeoDataFrame)
    assert isinstance(mcs_summary, gpd.GeoDataFrame), "has first and last centroid Points"
    assert 0 < len(mcs_summary) < len(mcs) < len(ce)

    assert mcs["nce"].eq(mcs.count_geometries()).all()  # codespell:ignore nce
    mcs.groupby("mcs_id").nce.mean().reset_index(drop=True).eq(  # codespell:ignore nce
        mcs_summary["mean_nce"]  # codespell:ignore nce
    ).all()
