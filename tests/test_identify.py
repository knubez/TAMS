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

    contours = [np.array([[0, 0], [1, 1], [0, 0.5]])]
    gdf = tams.core._contours_to_gdf(contours)
    assert len(contours) == 1
    assert len(gdf) == 1


def test_contour(msg_tb0, caplog):
    tb = msg_tb0

    with caplog.at_level("DEBUG", logger="tams"):
        cs_closed = tams.contour(tb, 219)

    debug_msgs = [r.message for r in caplog.records if r.levelname == "DEBUG"]
    assert set(debug_msgs) == {"skipping open contour"}

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
            "skipping an input contour with less than 2 points: []",
            id="empty",
        ),
        pytest.param(
            [[0, 0]],
            "skipping an input contour with less than 2 points: [[0 0]]",
            id="1-pt",
        ),
        pytest.param(
            [[0, 0], [0, 0]],
            "skipping invalid closed contour: An input LineString must be valid.",
            id="2-pt closed",
        ),
        pytest.param(
            [[0, 0], [1, 1]],
            "skipping open contour",
            id="2-pt open",
        ),
        pytest.param(
            [[0, 0], [1, 1], [0, 0]],
            "skipping invalid closed contour: Too few points in geometry component[0 0]",
            id="3-pt closed",
        ),
        pytest.param(
            [[0, 0], [1, 0], [0.5, 1], [0.5, -1], [0, 0]],
            "skipping invalid closed contour: Ring Self-intersection[0.5 0]",
            id="self-intersecting",
        ),
    ],
)
def test_contour_skipped(contour, messages, caplog):
    contours = [np.asarray(contour)]
    if isinstance(messages, str):
        messages = [messages]

    with caplog.at_level("DEBUG", logger="tams"):
        cs = tams.core._contours_to_gdf_new(contours)

    assert cs.empty

    debug_msgs = [r.message for r in caplog.records if r.levelname == "DEBUG"]
    assert debug_msgs == messages


@pytest.mark.parametrize(
    "contour, messages",
    [
        pytest.param(
            [[0, 0], [1, 1], [0, 0], [1, 1]],
            "skipping non-simple open contour",
            id="back-and-forth",
        ),
        pytest.param(
            [[0, 0], [1, 0], [0.5, 1], [0.5, -1]],
            "skipping non-simple open contour",
            id="self-intersecting (through)",
        ),
        pytest.param(
            [[0, 0], [1, 0], [0.5, 1], [0.5, 0]],
            "skipping non-simple open contour",
            id="self-intersecting (tangent)",
        ),
    ],
)
def test_contour_skipped_open(contour, messages, caplog):
    contours = [np.asarray(contour)]
    if isinstance(messages, str):
        messages = [messages]

    with caplog.at_level("DEBUG", logger="tams"):
        cs = tams.core._contours_to_gdf_new(contours, closed_only=False)

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

    cs = tams.core._contours_to_gdf_new(contours, closed_only=False)

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

    cs = tams.core._contours_to_gdf_new(contours)
    assert cs.empty

    cs = tams.core._contours_to_gdf_new(contours, tolerance=eps * 2)
    assert len(cs) == 1


def test_contour_tolerance_dedupe():
    eps = 1e-6
    contour = [[0, 0], [1, 0], [1, 0 + eps], [1, 1], [0, 1], [0, 0]]
    contours = [np.asarray(contour)]

    cs = tams.core._contours_to_gdf_new(contours)
    assert len(cs) == 1
    assert shapely.get_coordinates(cs.iloc[0].contour).shape[0] == len(contour)

    cs = tams.core._contours_to_gdf_new(contours, tolerance=eps * 2)
    assert len(cs) == 1
    assert shapely.get_coordinates(cs.iloc[0].contour).shape[0] == len(contour) - 1


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
