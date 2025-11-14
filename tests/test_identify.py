"""
Test :func:`tams.identify` and related routines.
"""

import numpy as np
import pytest
import shapely
import xarray as xr

import tams


def test_contour_too_small_skipped():
    # With a few of the sample MPAS data time steps (e.g. `.isel(time=22)`)
    # the current contouring algo returns some with less than 3 points,
    # which can't make a LinearRing (it raises ValueError)

    contours = [np.array([[0, 0], [1, 1]])]
    gdf = tams.core._contours_to_gdf(contours)
    assert len(contours) == 1
    assert len(gdf) == 0

    contours = [np.array([[0, 0], [1, 1], [0, 0.5]])]  # open
    gdf = tams.core._contours_to_gdf(contours)
    assert len(contours) == 1
    assert len(gdf) == 0

    contours = [np.array([[0, 0], [1, 1], [0, 0.5], [0, 0]])]  # closed
    gdf = tams.core._contours_to_gdf(contours)
    assert len(contours) == 1
    assert len(gdf) == 1


def test_contour(msg_tb0, caplog):
    tb = msg_tb0

    with caplog.at_level("DEBUG", logger="tams"):
        cs_closed = tams.contour(tb, 219)

    debug_msgs = [r.message for r in caplog.records if r.levelname == "DEBUG"]
    # assert set(debug_msgs) == {"skipping open contour"}
    assert debug_msgs == ["skipped 2 open contours"]

    assert set(cs_closed.geom_type) == {"LinearRing"}
    assert cs_closed.closed.all()  # our series
    assert cs_closed.is_closed.all()  # gpd property

    caplog.clear()
    with caplog.at_level("DEBUG", logger="tams"):
        cs = tams.contour(tb, 219, closed_only=False)

    debug_msgs = [r.message for r in caplog.records if r.levelname == "DEBUG"]
    assert not debug_msgs

    assert (~cs.closed).sum() == (~cs.is_closed).sum() == (len(cs) - len(cs_closed)) == 2

    for gdf in [cs_closed, cs]:
        assert gdf.columns.tolist() == ["contour", "closed", "encloses_higher"]
        assert gdf.active_geometry_name == "contour"
        assert gdf.crs == "EPSG:4326"
        assert set(gdf.geom_type) <= {"LinearRing", "LineString"}
        assert gdf.dtypes["contour"] == "geometry"
        assert gdf.dtypes["closed"] == bool
        assert gdf.dtypes["encloses_higher"] == "boolean"


@pytest.mark.parametrize(
    "contour, messages",
    [
        pytest.param(
            [[]],
            "skipped contours: too few points: 1",
            id="empty",
        ),
        pytest.param(
            [[0, 0]],
            "skipped contours: too few points: 1",
            id="1-pt",
        ),
        pytest.param(
            [[0, 0], [0, 0]],
            "skipped contours: invalid closed (An input LineString must be valid.): 1",
            id="2-pt closed",
        ),
        pytest.param(
            [[0, 0], [1, 1]],
            "skipped contours: open: 1",
            id="2-pt open",
        ),
        pytest.param(
            [[0, 0], [1, 1], [0, 0]],
            "skipped contours: invalid closed (Too few points in geometry component[0 0]): 1",
            id="3-pt closed",
        ),
        pytest.param(
            [[0, 0], [1, 0], [0.5, 1], [0.5, -1], [0, 0]],
            "skipped contours: invalid closed (Ring Self-intersection[0.5 0]): 1",
            id="self-intersecting",
        ),
    ],
)
def test_contour_skipped(contour, messages, caplog):
    contours = [np.asarray(contour)]
    if isinstance(messages, str):
        messages = [messages]

    with caplog.at_level("DEBUG", logger="tams"):
        cs = tams.core._contours_to_gdf(contours)

    assert cs.empty

    debug_msgs = [r.message for r in caplog.records if r.levelname == "DEBUG"]
    assert debug_msgs == messages


@pytest.mark.parametrize(
    "contour, messages",
    [
        pytest.param(
            [[0, 0], [1, 1], [0, 0], [1, 1]],
            "skipped contours: non-simple open: 1",
            id="back-and-forth",
        ),
        pytest.param(
            [[0, 0], [1, 0], [0.5, 1], [0.5, -1]],
            "skipped contours: non-simple open: 1",
            id="self-intersecting (through)",
        ),
        pytest.param(
            [[0, 0], [1, 0], [0.5, 1], [0.5, 0]],
            "skipped contours: non-simple open: 1",
            id="self-intersecting (tangent)",
        ),
    ],
)
def test_contour_skipped_open(contour, messages, caplog):
    contours = [np.asarray(contour)]
    if isinstance(messages, str):
        messages = [messages]

    with caplog.at_level("DEBUG", logger="tams"):
        cs = tams.core._contours_to_gdf(contours, closed_only=False)

    assert cs.empty

    debug_msgs = [r.message for r in caplog.records if r.levelname == "DEBUG"]
    assert debug_msgs == messages


def test_contour_repeated_point():
    closed_contour = [[0, 0], [1, 0], [1, 0], [0.5, 1], [0.5, 1], [0, 0]]
    open_contour = [[0, 0], [0, 0], [1, 1], [2, 2], [2, 2]]
    contours = [
        np.asarray(closed_contour),
        np.asarray(open_contour),
    ]
    assert [len({tuple(row) for row in c}) for c in contours] == [3] * len(contours)

    cs = tams.core._contours_to_gdf(contours, closed_only=False)

    assert len(cs) == 2
    assert cs.iloc[0].closed
    assert not cs.iloc[1].closed

    for geom in cs.geometry:
        assert geom.is_valid
        assert geom.is_simple
        assert shapely.get_coordinates(geom).shape[0] == 4 if geom.is_closed else 3


def test_contour_tolerance_snap():
    eps = 1e-6
    contour = [[0, 0], [1, 0], [1, 1], [0, 1], [0, 0 + eps]]
    contours = [np.asarray(contour)]

    cs = tams.core._contours_to_gdf(contours)
    assert cs.empty

    cs = tams.core._contours_to_gdf(contours, tolerance=eps * 2)
    assert len(cs) == 1


def test_contour_tolerance_dedupe():
    eps = 1e-6
    contour = [[0, 0], [1, 0], [1, 0 + eps], [1, 1], [0, 1], [0, 0]]
    contours = [np.asarray(contour)]

    cs = tams.core._contours_to_gdf(contours)
    assert len(cs) == 1
    assert shapely.get_coordinates(cs.iloc[0].contour).shape[0] == len(contour)

    cs = tams.core._contours_to_gdf(contours, tolerance=eps * 2)
    assert len(cs) == 1
    assert shapely.get_coordinates(cs.iloc[0].contour).shape[0] == len(contour) - 1


@pytest.mark.parametrize("rem", [0, 1], ids=["inner-ccw", "inner-cw"])
@pytest.mark.parametrize("rev", [True, False], ids=["rev", "fwd"])
@pytest.mark.parametrize("num", [4, 5])
def test_contours_to_shields_auto_edge(rem, rev, num):
    # Nested contours, auto detecting if polygon edges are `encloses_higher`
    # true or false

    # CCW square made of ones
    inner_ccw = rem == 0
    is_even = num % 2 == 0
    is_odd = not is_even
    ones_square = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1], [-1, -1]])

    # Scale the square, reversing winding every other
    segs = [f * ones_square[:: -1 if f % 2 == rem else 1] for f in range(1, num + 1)]
    i_inner, i_outer = 0, -1
    if rev:
        segs = segs[::-1]
        i_outer, i_inner = i_inner, i_outer
    assert shapely.LinearRing(segs[i_inner]).is_ccw == inner_ccw

    cs = tams.core._contours_to_gdf(segs)
    assert len(cs) == len(segs)
    assert cs.iloc[i_outer].encloses_higher == (not inner_ccw if is_even else inner_ccw)
    assert cs.encloses_higher.sum() == len(cs) // 2 + int(is_odd and inner_ccw)

    el = tams.core._contours_to_shields(cs)
    assert el.area.is_monotonic_increasing
    assert len(el) == len(cs) // 2 + int(is_odd)
    if is_even:
        assert el.geometry.interiors.str.len().eq(1).all()
    else:
        assert len(el.iloc[0].geometry.interiors) == 0
        assert el.iloc[1:].geometry.interiors.str.len().eq(1).all()


def test_contours_to_shields_manual_edge():
    # Two contours, one inside the other

    # CCW square made of ones
    ones_square = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1], [-1, -1]])
    segs = [(2 * ones_square), ones_square[::-1]]

    assert shapely.LinearRing(segs[0]).is_ccw
    assert not shapely.LinearRing(segs[1]).is_ccw

    cs = tams.core._contours_to_gdf(segs)
    assert len(cs) == len(segs)
    assert cs.iloc[0].encloses_higher  # edge
    assert not cs.iloc[1].encloses_higher

    # Outermost encloses higher, but we specified edges don't enclose higher
    # so the outermost is a hole with no parent (not returned),
    # and the innermost is a polygon with no holes
    el = tams.core._contours_to_shields(cs, edge_encloses_higher=False)
    assert len(el) == 1
    assert el.geometry.interiors.str.len().eq(0).all()
    assert (shapely.get_coordinates(el.iloc[0].geometry) == segs[1][::-1]).all()

    # Outermost encloses higher, and we specified edges do enclose higher
    # so we get one polygon with one hole
    el = tams.core._contours_to_shields(cs, edge_encloses_higher=True)
    assert len(el) == 1
    assert el.geometry.interiors.str.len().eq(1).all()
    assert (shapely.get_coordinates(el.iloc[0].geometry.exterior) == segs[0]).all()
    assert (shapely.get_coordinates(el.iloc[0].geometry.interiors[0]) == segs[1]).all()


def test_contours_to_shields_two_holes():
    # CCW square made of ones
    ones_square = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1], [-1, -1]])
    h1 = ones_square[::-1] - np.r_[1.5, 0]
    h2 = ones_square[::-1] + np.r_[1.5, 0]
    segs = [(5 * ones_square), h1, h2]

    cs = tams.core._contours_to_gdf(segs)
    assert len(cs) == len(segs)

    el = tams.core._contours_to_shields(cs)
    assert len(el) == 1
    assert el.geometry.interiors.str.len().eq(2).all()


def test_identify_no_ces_warning(msg_tb0):
    tb = msg_tb0
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
