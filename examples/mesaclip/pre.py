"""
Streamline data
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import xarray as xr

IN_BASE = Path("/glade/derecho/scratch/fudan/MOAAP/results")
IN_BASE_MOD = IN_BASE / "CESM-HR"
IN_BASE_OBS = IN_BASE / "OBS"

# Example paths (first)
IN_MOD_EX = IN_BASE_MOD / "200001_CESM-HR_ObjectMasks__dt-1h_MOAAP-masks.nc"
IN_OBS_EX = IN_BASE_OBS / "200101_ERA5_ObjectMasks__dt-1h_MOAAP-masks.nc"


def get_years_files(dir: Path) -> dict[int, list[Path]]:
    d = defaultdict(list)
    for p in sorted(dir.glob("*.nc")):
        s_ym, *_ = p.stem.split("_")
        year = int(s_ym[:4])
        d[year].append(p)
    return d


FILES = {
    "mod": get_years_files(IN_BASE_MOD),
    "obs": get_years_files(IN_BASE_OBS),
}


def preprocess(ds: xr.Dataset) -> xr.Dataset:
    p = Path(ds.encoding["source"])

    _, which, *_ = p.stem.split("_")
    if which == "CESM-HR":
        is_mod = True
    elif which == "ERA5":
        is_mod = False
    else:
        raise ValueError(f"Unrecognized path name: {p.name!r}")
    is_obs = not is_mod

    # For obs, need to flip y (ERA5)
    if is_obs:
        ds = ds.sel(yc=slice(None, None, -1))

    # lat/lon are really 1-d
    lon0, lat0 = ds.lon.isel(yc=0), ds.lat.isel(xc=0)
    assert (lon0.diff("xc") == 0.25).all()
    assert (lat0.diff("yc") == 0.25).all()
    assert (ds.lon == lon0).all()
    assert (ds.lat == lat0).all()
    ds = ds.assign(lat=lat0, lon=lon0).swap_dims(xc="lon", yc="lat")

    # Select vars we want and spatial region
    ds = (
        ds[["BT", "PR"]]
        .sel(lat=slice(-70, 75) if is_mod else slice(-60, 60))
        .rename(
            {
                "BT": "tb",
                "PR": "pr",
            }
        )
    )

    # Variable attrs
    ds["tb"].attrs = {
        "long_name": "brightness temperature",
        "units": "K",
    }
    ds["pr"].attrs = {
        "long_name": "precipitation rate",
        "units": "mm/hr",
    }

    if is_obs:
        # Try to fill in the null brightness temp pixels a bit
        # Can't use the HH:30 time to help since not in the dataset
        # n_na0 = ds["tb"].isnull().sum(dim=("lat", "lon"))
        ds["tb"] = (
            ds["tb"]
            .interpolate_na(
                dim="lat",
                method="nearest",
                fill_value="extrapolate",
                assume_sorted=True,
            )
            .interpolate_na(
                dim="lon",
                method="nearest",
                fill_value="extrapolate",
                assume_sorted=True,
            )
            # .interpolate_na(
            #     dim="time",
            #     method="linear",
            #     max_gap="1h",
            #     assume_sorted=True,
            # )
        )
        # n_na = ds["tb"].isnull().sum(dim=("lat", "lon"))
        # assert n_na.sum() <= n_na0.sum()

    ds.attrs = {
        "case": "mod" if is_mod else "obs",
    }

    return ds


def load_path(p: Path) -> xr.Dataset:
    return preprocess(xr.open_dataset(p))


def load_year(files: list[Path]) -> xr.Dataset:
    ds = xr.open_mfdataset(
        files,
        preprocess=preprocess,
        combine="nested",
        concat_dim="time",
        chunks={"time": 1, "lat": -1, "lon": -1},
    )

    # Model is no-leap, so normalize to that
    if ds.attrs["case"] == "mod":
        assert ds.sizes["time"] == 365 * 24  # always
    elif ds.attrs["case"] == "obs":
        if ds.time.isel(time=0).dt.is_leap_year:
            assert ds.sizes["time"] == 366 * 24
            # Feb 29 will be dropped automatically
        else:
            assert ds.sizes["time"] == 365 * 24
    else:
        raise AssertionError
    ds = ds.convert_calendar("365_day")

    return ds


m = load_year(FILES["mod"][2000])
o = load_year(FILES["obs"][2001])


from dask.diagnostics import ProgressBar

print("mod")
with ProgressBar():
    m.to_netcdf("mod.nc")

print("obs")
with ProgressBar():
    o.to_netcdf("obs.nc")
