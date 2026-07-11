"""
Test routines from :mod:`tams.data`.
"""

import re
from typing import get_overloads

import pytest

import tams

from . import skipif_no_earthdata


def test_load_pooch_missing(mocker):
    mocker.patch.dict("sys.modules", pooch=None)
    with pytest.raises(RuntimeError, match="pooch is required"):
        _ = tams.data.open_example("msg-rad")


def test_load_gdown_missing(mocker, tmpdir):
    # Note gdown isn't needed/used if the file is already cached
    mocker.patch.dict("sys.modules", gdown=None)
    with (
        tams.set_options(cache_location=tmpdir),
        pytest.raises(RuntimeError, match="gdown is required"),
    ):
        _ = tams.data.open_example("msg-rad")


def test_load_msg_tb_sample(msg_tb0):
    tb = msg_tb0
    assert tb.name == "tb"
    assert tb.attrs["channel"] == 9
    assert tuple(tb.coords) == ("lon", "lat", "time")


def test_load_mpas_sample(mpas):
    ds = mpas
    assert tuple(ds.data_vars) == ("tb", "pr")
    assert tuple(ds.coords) == ("time", "lon", "lat")


@pytest.mark.parametrize("func_name", ["open_example", "load_example"])
def test_example_overload_annotations_match_registry(func_name):
    func = getattr(tams.data, func_name)
    overloads = get_overloads(func)

    assert len(overloads) == 2

    public_lut = {
        **tams.data._EXAMPLE_FILE_DIRECT_LUT,
        **tams.data._EXAMPLE_FILE_INDIRECT_LUT,
    }
    expected_nc_keys = {k for k, f in public_lut.items() if f.file_type.is_nc}
    expected_parquet_keys = {
        k for k, f in tams.data._EXAMPLE_FILE_DIRECT_LUT.items() if not f.file_type.is_nc
    }

    observed = {}
    for overload in overloads:
        key_annotation = overload.__annotations__["key"]
        return_annotation = overload.__annotations__["return"]
        keys = set(re.findall(r'["\']([^"\']+)["\']', key_annotation))
        observed[return_annotation] = keys

    assert observed["xarray.Dataset"] == expected_nc_keys
    assert observed["geopandas.GeoDataFrame"] == expected_parquet_keys


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
