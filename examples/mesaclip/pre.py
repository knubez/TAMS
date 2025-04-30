#!/usr/bin/env python
"""
Streamline data
"""

from __future__ import annotations

import os
import subprocess
from collections import defaultdict
from pathlib import Path

import geopandas as gpd
import pandas as pd
import xarray as xr
from joblib import Parallel, delayed

import tams

IN_BASE = Path("/glade/derecho/scratch/fudan/MOAAP/results")
IN_BASE_MOD = IN_BASE / "CESM-HR"
IN_BASE_OBS = IN_BASE / "OBS"

HERE = Path(__file__).parent
REPO = HERE.parent.parent
assert REPO.name == "TAMS"
OUT = Path("/glade/derecho/scratch/zmoon/mesaclip")

# Note GeoPandas v1 required to use these options
GP_ENCODING = "geoarrow"
GP_SCHEMA_VERSION = "1.1.0"

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
    assert ds.sizes["time"] == 365 * 24

    return ds


NC_ENCODING = {
    "tb": {"zlib": True, "complevel": 3},
    "pr": {"zlib": True, "complevel": 3},
}


def add_ce_stats(pr: xr.DataArray, ce: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Add time and precip stats to CE frame."""
    t = pr.time.item()
    ts = pd.to_datetime(str(t))  # pretend it's normal, instead of 365-day
    ce = ce.drop(columns=["inds219", "area219_km2", "cs219"]).assign(time=ts)

    # Add precip stats needed for the special classify
    ce = tams.data_in_contours(
        pr,
        ce,
        agg=("count", "mean"),
        merge=True,
    )
    ce = tams.data_in_contours(
        pr.where(pr >= 2).rename("pr_ge_2"),
        ce,
        agg=("count", "mean"),
        merge=True,
    )

    return ce


def preprocess_year_ds(ds: xr.Dataset, *, parallel: bool = True) -> xr.Dataset:
    """Preprocess a year of data by month, saving CE GeoParquet files."""
    assert ds.time.dt.year.to_series().nunique() == 1
    year = ds.time.dt.year.values[0]

    for month, g in ds.groupby("time.month"):
        ym = f"{year:04d}-{month:02d}"

        ces0, _ = tams.identify(g.tb, parallel=parallel)

        if parallel:
            ces = Parallel(n_jobs=-2, verbose=10)(
                delayed(add_ce_stats)(g.pr.isel(time=i).copy(deep=False), ce0.copy())
                for i, ce0 in enumerate(ces0)
            )
        else:
            ces = [add_ce_stats(g.pr.isel(time=i), ce0) for i, ce0 in enumerate(ces0)]

        gdf = pd.concat(ces, ignore_index=True)
        gdf.attrs = {
            "case": ds.attrs["case"],
        }
        gdf.to_parquet(
            OUT / "ce" / f"{ds.attrs['case'][0]}{ym}.parquet",
            geometry_encoding=GP_ENCODING,
            schema_version=GP_SCHEMA_VERSION,
        )

    return ces0


JOB_TPL_PRE = r"""
#/bin/bash
## Submit with `qsub -A <account>`
#PBS -N mesaclip1
#PBS -q casper
#PBS -l walltime=2:00:00
#PBS -l select=1:ncpus=21:mem=80gb
#PBS -j oe

cd /glade/u/home/zmoon/git/TAMS/examples/mesaclip

py=/glade/u/home/zmoon/mambaforge/envs/tams/bin/python

$py -c "from pre import FILES, load_year, preprocess_year_ds;
preprocess_year_ds(load_year(FILES[{which!r}][{year}]))"
""".lstrip()


def submit_pres():
    A = os.getenv("A")
    if A is None:
        print("set $A to desired account")
        raise SystemExit(2)
    for which, years in FILES.items():
        for year, _ in years.items():
            job = JOB_TPL_PRE.format(which=which, year=year)
            job_file = REPO / f"mesaclip1_{which}_{year}.sh"
            with open(job_file, "w") as f:
                f.write(job)
            print(f"Submitting {job_file}")
            subprocess.run(["qsub", "-A", A, str(job_file)], check=True)
            break


if __name__ == "__main__":
    submit_pres()
