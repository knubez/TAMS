"""
Test :func:`tams.identify` and related routines.
"""

import numpy as np
import pytest
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
            "skipping invalid closed contour: A linearring requires at least 4 coordinates.",
            id="2-pt open",
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
