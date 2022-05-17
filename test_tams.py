from pathlib import Path

import numpy as np
import pytest

import tams

r = tams.load_example_ir().isel(time=0)

tb = tams.tb_from_ir(r, ch=9)

glade_avail = Path("/glade").is_dir()


def test_ch9_tb_loaded():
    assert tb.name == "ch9"
    assert tuple(tb.coords) == ("lon", "lat", "time")


def test_data_in_contours_methods_same_result():
    cs235 = tams._contours_to_gdf(tams.contours(tb, 235))
    cs219 = tams._contours_to_gdf(tams.contours(tb, 219))
    cs235, _ = tams._size_filter_contours(cs235, cs219)

    varnames = ["ch9"]
    x1 = tams._data_in_contours_sjoin(tb, cs235, varnames=varnames)
    x2 = tams._data_in_contours_regionmask(tb, cs235, varnames=varnames)
    assert (x1.count_ch9 > 0).all()
    assert x1.count_ch9.equals(x2.count_ch9)
    assert x1.std_ch9.equals(x2.std_ch9)
    # TODO: mean values aren't exactly the same


def test_data_in_contours_non_xy():
    # The MPAS one has real lat/lon 1-D dim-coords, not x/y with 2-D lat/lon
    # so with `to_dataframe().reset_index(drop=True)` lat/lon were lost
    ds = tams.load_example_mpas().isel(time=1)
    cs = tams.identify(ds.tb)[0][0]
    data = ds.precip
    cs_precip = tams.data_in_contours(data, cs, method="sjoin")
    assert cs_precip.mean_precip.sum() > 0
    assert (cs_precip.count_precip > 0).all()


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
    ell = Ellipse((1, 1), w, h, np.rad2deg(np.pi / 4))
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
    gdf = tams._contours_to_gdf(contours)
    assert len(contours) == 1
    assert len(gdf) == 0

    contours = [np.array([[0, 0], [1, 1], [0, 0.5]])]
    gdf = tams._contours_to_gdf(contours)
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
    assert ds.dims["time"] == 24
