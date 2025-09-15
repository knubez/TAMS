"""
Test :func:`tams.run`.
"""

import geopandas as gpd
import xarray as xr

import tams


def test_tams_run_basic(mpas):
    # Keep this like the notebook example but can be shorter
    ds = mpas.rename({"tb": "ctt"}).isel(time=slice(1, 7))
    assert isinstance(ds, xr.Dataset)

    ce, mcs, mcs_summary = tams.run(ds)
    assert isinstance(ce, gpd.GeoDataFrame)
    assert isinstance(mcs, gpd.GeoDataFrame)
    assert isinstance(mcs_summary, gpd.GeoDataFrame), "has first and last centroid Points"
    assert 0 < len(mcs_summary) < len(mcs) < len(ce)

    assert mcs["nce"].eq(mcs.count_geometries()).all()  # codespell:ignore nce
    mcs.groupby("mcs_id").nce.mean().reset_index(drop=True).eq(  # codespell:ignore nce
        mcs_summary["mean_nce"]  # codespell:ignore nce
    ).all()


# TODO: test to check crs of all geometry columns returned by `run` are correct
# and all have a RangeIndex
