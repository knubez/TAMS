"""
Streamline data
"""

from pathlib import Path

import xarray as xr

IN_BASE = Path("/glade/derecho/scratch/fudan/MOAAP/results")
IN_BASE_MOD = IN_BASE / "CESM-HR"
IN_BASE_OBS = IN_BASE / "OBS"

# Example paths (first)
IN_MOD_EX = IN_BASE_MOD / "200001_CESM-HR_ObjectMasks__dt-1h_MOAAP-masks.nc"
IN_OBS_EX = IN_BASE_OBS / "200101_ERA5_ObjectMasks__dt-1h_MOAAP-masks.nc"


def load_path(p: Path) -> xr.Dataset:
    which = p.stem.split("_")
    if which == "CESM-HR":
        is_mod = True
    elif which == "ERA5":
        is_mod = False
    else:
        raise ValueError(f"Unrecognized path name: {p.name!r}")
    is_obs = not is_mod

    ds = xr.open_dataset(p)

    ds = (
        ds[["BT", "PR"]]
        .sel(yc=slice(-70, 75) if is_mod else slice(-60, 60))
        .rename(
            {
                "BT": "tb",
                "PR": "pr",
            }
        )
    )

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

    if is_obs:
        # Try to fill in the null brightness temp pixels a bit
        # Can't use the HH:30 time to help since not in the dataset
        n_na0 = ds["tb"].isnull().sum()
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
            .interpolate_na(
                dim="time",
                method="linear",
                max_gap="1h",
                assume_sorted=True,
            )
        )
        n_na = ds["tb"].isnull().sum()
        assert n_na < n_na0 or n_na == n_na0 == 0
        ds["tb"].attrs.update(
            {
                "_n_na_pre_interp": n_na0,
                "_n_na": n_na,
            }
        )

    # TODO: set calendar to non-leap with cftime?
    # And drop leap day data from obs, which seems to have it
    # OR interp leap day for the model

    return ds
