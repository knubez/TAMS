"""
Test routines from :mod:`tams.data`.
"""

import re
from pathlib import Path
from typing import get_overloads

import pytest

import tams

from . import skipif_no_earthdata

EF = tams.data._ExampleFile
EFT = tams.data._ExampleFileType


def _public_lut() -> dict[str, EF]:
    return {
        **tams.data._EXAMPLE_FILE_DIRECT_LUT,
        **tams.data._EXAMPLE_FILE_INDIRECT_LUT,
    }


def _public_current_keys() -> set[str]:
    public_lut = _public_lut()
    keys = {k for k in public_lut if re.search(r"-v\d+(?:\.\d+)*$", k) is None}
    assert keys, "No current keys identified in the public LUT"
    return keys


def _api_rst_example_table_keys() -> set[str]:
    p = Path(__file__).resolve().parents[1] / "docs" / "api.rst"
    text = p.read_text(encoding="utf-8")

    try:
        section = text.split(".. _example_datasets:", maxsplit=1)[1]
        section = section.split("External data sources", maxsplit=1)[0]
    except IndexError as e:
        raise AssertionError("Could not locate example-datasets table in docs/api.rst") from e

    keys = set()
    for line in section.splitlines():
        stripped = line.strip()
        if not stripped.startswith("* - ``"):
            continue

        code_keys = re.findall(r"``([^`]+)``", stripped)
        if "..." in stripped and len(code_keys) == 2:
            first, last = code_keys
            prefix1, n1 = first.rsplit("-", maxsplit=1)
            prefix2, n2 = last.rsplit("-", maxsplit=1)
            if prefix1 != prefix2:
                raise AssertionError(f"Unexpected ellipsis key range: {stripped!r}")
            for i in range(int(n1), int(n2) + 1):
                keys.add(f"{prefix1}-{i}")
        else:
            keys.update(code_keys)

    return keys


def _data_module_example_keys() -> set[str]:
    keys = set()
    for func_name in ("fetch_example", "open_example", "load_example"):
        func = getattr(tams.data, func_name)
        doc = func.__doc__ or ""
        found = re.findall(r'tams\.data\.(?:fetch|open|load)_example\("([^"]+)"\)', doc)
        keys.update(found)
    return keys


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

    public_lut = _public_lut()
    expected_nc_keys = {k for k, f in public_lut.items() if f.file_type is EFT.NETCDF}
    expected_geoparquet_keys = {k for k, f in public_lut.items() if f.file_type is EFT.GEOPARQUET}
    expected_parquet_keys = {k for k, f in public_lut.items() if f.file_type is EFT.PARQUET}

    observed = {}
    for overload in overloads:
        key_annotation = overload.__annotations__["key"]
        return_annotation = overload.__annotations__["return"]
        keys = set(re.findall(r'["\']([^"\']+)["\']', key_annotation))
        observed[return_annotation] = keys

    assert len(overloads) == 2 + int(bool(expected_parquet_keys))
    assert observed["xarray.Dataset"] == expected_nc_keys
    assert observed["geopandas.GeoDataFrame"] == expected_geoparquet_keys
    if expected_parquet_keys:
        assert observed["pandas.DataFrame"] == expected_parquet_keys
    else:
        assert "pandas.DataFrame" not in observed


def test_api_doc_example_table_up_to_date():
    expected = _public_current_keys()
    observed = _api_rst_example_table_keys()
    assert observed == expected


def test_docstring_examples_current():
    valid = _public_current_keys()
    observed = _data_module_example_keys()
    assert observed, "No tams.data.*_example(...) calls found in data-module docstring examples"
    assert observed <= valid


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
