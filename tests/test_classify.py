"""
Test :func:`tams.classify`.
"""

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest

import tams


def test_classify_empty():
    cs = gpd.GeoDataFrame(
        columns=["mcs_id", "geometry", "time", "dtime", "area_km2", "area_core_km2"],
        crs="EPSG:4326",
    )
    with pytest.warns(UserWarning, match="empty input frame"):
        cs_ = tams.classify(cs)
        assert "mcs_class" in cs_ and "mcs_class" not in cs


def test_classify_cols_check():
    cs = gpd.GeoDataFrame(
        columns=["mcs_id", "geometry", "time", "area_km2", "area_core_km2"],
        data=[[0, None, pd.NaT, np.nan, np.nan]],
        crs="EPSG:4326",
    )
    with pytest.raises(
        ValueError,
        match=r"missing these columns needed by the classify algorithm: \['dtime'\]",
    ):
        _ = tams.classify(cs)
