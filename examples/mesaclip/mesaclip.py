#!/usr/bin/env python
"""
Run TAMS for MESACLIP, using the MOAAP output files.
"""

from __future__ import annotations

import os
import subprocess
import time
from collections import defaultdict
from io import StringIO
from pathlib import Path
from typing import Literal, NamedTuple, Self

import geopandas as gpd
import numpy as np
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
    """Find and sort the mod/obs input files."""
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

FILL_TB = {
    "mod": {
        2006: [0],  # 2006-01-01 00:00
        2090: [1],  # 2090-01-01 01:00
    },
    "obs": {
        2002: [6441, 7168],  # 2002-09-26 09:00, 2002-10-26 16:00
        2007: [2284, 2285, 2286, 2287],  # 2007-04-06 04:00 through 07:00
        2022: [8, 709],  # 2022-01-01 08:00, 2022-01-30 13:00
    },
}


def fill_time(ds: xr.Dataset, i: int, *, vn="tb", method="linear") -> xr.Dataset:
    """Fill variable `vn` at time `i` in-place but lazily."""
    n = ds.sizes["time"]
    assert n >= 3
    assert 0 <= i < n

    if i == 0:
        a, b = i, i + 2
        j = 0
    elif i == n - 1:
        a, b = i - 2, i
        j = 2
    else:
        a, b = i - 1, i + 1
        j = 1

    da = ds[vn].isel(time=slice(a, b + 1))
    assert da.sizes["time"] == 3
    da = da.where(da > 0)

    new = (
        da.chunk({"time": -1})
        .interpolate_na(
            dim="time",
            method=method,
            # max_gap="1h",  # not working with cftime?
            fill_value="extrapolate",
            keep_attrs=True,
        )
        .isel(time=j)
        .assign_attrs(i=i)
    )

    ds[vn][{"time": i}] = new

    return ds


def preprocess(ds: xr.Dataset) -> xr.Dataset:
    """Select tb and pr and clean up things."""
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

    assert ds.time.dt.year.to_series().nunique() == 1
    year = ds.time.dt.year.values[0]

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
                keep_attrs=True,
            )
            .interpolate_na(
                dim="lon",
                method="nearest",
                fill_value="extrapolate",
                assume_sorted=True,
                keep_attrs=True,
            )
            # .interpolate_na(
            #     dim="time",
            #     method="linear",
            #     max_gap="1h",
            #     assume_sorted=True,
            #     keep_attrs=True,
            # )
        )
        # n_na = ds["tb"].isnull().sum(dim=("lat", "lon"))
        # assert n_na.sum() <= n_na0.sum()
    else:
        inds_to_fill = FILL_TB["mod"].get(year, [])
        for i in inds_to_fill:
            ds = fill_time(ds, i, vn="tb", method="linear")

    ds.attrs = {
        "case": "mod" if is_mod else "obs",
    }

    return ds


def load_path(p: Path) -> xr.Dataset:
    """Load a single mod/obs input file, invoking the preprocess routine."""
    return preprocess(xr.open_dataset(p))


def load_year(files: list[Path]) -> xr.Dataset:
    """Load a year of mod/obs input files with Dask, invoking the preprocess routine
    for each one and converting calendar to 365-day, for obs consistency with mod."""
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


def find_null_tb(ds: xr.Dataset) -> xr.Dataset:
    """Find times of all-null Tb."""

    assert ds.time.dt.year.to_series().nunique() == 1
    year = ds.time.dt.year.values[0]
    case = ds.attrs["case"]
    print(f"Searching for null Tb in {case} {year}")

    def fun(da):
        assert da.dims == ("lat", "lon")
        return da.isnull().all().compute().item()

    # Use joblib (single-time chunks)
    is_nulls = Parallel(n_jobs=-2, verbose=10)(
        delayed(fun)(ds["tb"].isel(time=i)) for i in range(ds.sizes["time"])
    )

    if not any(is_nulls):
        return

    # Save data
    with open(HERE / f"null_tb_{case}_{year}.txt", "w") as f:
        for i, is_null in enumerate(is_nulls):
            if is_null:
                ts = str(ds.time.values[i])
                print(f"Null Tb at time step {i} ({ts})")
                f.write(f"{i},{ts}\n")


JOB_TPL_NULL_TB = r"""
#!/bin/bash
#PBS -N null-tb
#PBS -q casper
#PBS -l walltime=2:00:00
#PBS -l select=1:ncpus=21:mem=80gb
#PBS -j oe
#PBS -d /glade/u/home/zmoon/git/TAMS/examples/mesaclip

py=/glade/u/home/zmoon/mambaforge/envs/tams/bin/python

$py -c "from mesaclip import FILES, load_year, find_null_tb
find_null_tb(load_year(FILES[{which!r}][{year}]))"
""".lstrip()


def get_account() -> str:
    """Get the account name for qsub from environment variable A.

    Raises
    ------
    SystemExit
        If A is not set in the environment, we exit with error code 2.
    """
    A = os.getenv("A")
    if A is None:
        print("set $A to desired account")
        raise SystemExit(2)
    return A


def submit_job(job: str, *, stem: str = "job") -> None:
    """Submit a job to PBS."""
    job_file = HERE / f"{stem}.sh"
    with open(job_file, "w") as f:
        f.write(job)
    print(f"Submitting {job_file}")
    subprocess.run(["qsub", "-A", get_account(), str(job_file)], check=True)


def submit_null_tb():
    """Find null Tb times in the obs input files."""
    for which, years in FILES.items():
        if which == "mod":
            continue
        for year, _ in years.items():
            job = JOB_TPL_NULL_TB.format(which=which, year=year)
            stem = f"null_tb_{which}_{year}"
            submit_job(job, stem=stem)


def add_ce_stats(pr: xr.DataArray, ce: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Add time and precip stats to CE frame."""
    t = pr.time.item()
    ts = pd.to_datetime(str(t))  # pretend it's normal, instead of 365-day
    ce = ce.drop(columns=["inds219", "area219_km2", "cs219"]).assign(time=ts)

    # Some obs times don't yield CEs even though they have Tb, e.g.
    # - 2003-09-19 14:00 (i=446)
    if ce.empty:
        for vn in [
            "pr_count",
            "pr_mean",
            "pr_ge_2_count",
            "pr_ge_2_mean",
        ]:
            ce[vn] = np.nan
        return ce

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
        ym = f"{year:04d}{month:02d}"
        w = ds.attrs["case"][0]  # which

        ces0, _ = tams.identify(g.tb, parallel=parallel)  # FIXME: thresholds

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
            OUT / "ce" / f"{w}{ym}.parquet",
            geometry_encoding=GP_ENCODING,
            schema_version=GP_SCHEMA_VERSION,
        )

    return ces0


JOB_TPL_PRE = r"""
#!/bin/bash
#PBS -N mesaclip1
#PBS -q casper
#PBS -l walltime=2:00:00
#PBS -l select=1:ncpus=21:mem=80gb
#PBS -j oe
#PBS -d /glade/u/home/zmoon/git/TAMS/examples/mesaclip

py=/glade/u/home/zmoon/mambaforge/envs/tams/bin/python

$py -c "from mesaclip import FILES, load_year, preprocess_year_ds;
preprocess_year_ds(load_year(FILES[{which!r}][{year}]))"
""".lstrip()


def submit_pres():
    """Submit jobs to preprocess the mod/obs input files."""
    for which, years in FILES.items():
        for year, _ in years.items():
            job = JOB_TPL_PRE.format(which=which, year=year)
            stem = f"mesaclip1_{which}_{year}"
            submit_job(job, stem=stem)


class Case(NamedTuple):
    """Case to track."""

    which: Literal["mod", "obs"]
    period: Literal["present", "future"]

    def from_path(p: Path) -> Self:
        """Get case from path."""
        if p.name.startswith("m"):
            which = "mod"
        elif p.name.startswith("o"):
            which = "obs"
        else:
            raise ValueError(f"Unrecognized path name: {p.name!r}")

        year = int(p.stem[1:5])
        if year < 2030:
            period = "present"
        else:
            period = "future"

        return Case(which, period)

    def to_id(self, *, concise: bool = True) -> str:
        """A string case ID."""
        if concise:
            return f"{self.which[0]}{self.period[0]}"
        else:
            return f"{self.which}_{self.period}"


def get_pre_files():
    """Get and sort the files to track, using :class:`Case` instances as keys."""

    out = defaultdict(list)
    files = sorted((OUT / "ce").glob("*.parquet"))
    for p in files:
        case = Case.from_path(p)
        out[case].append(p)

    return out


def check_pre_files():
    """Check that we have a preprocessed CE file for each of the expected YYYYMMs.

    Raises
    ------
    AssertionError
        If any of the expected YYYYMMs are missing.
        Info included in the exception message.
    """

    def get_ym(p: Path) -> str:
        return p.stem[1:7]

    ok = True
    s = StringIO()
    for case, files in get_pre_files().items():
        s.write(f"{case!r}\n")

        # First is Jan
        stem = files[0].stem
        if not stem.endswith("01"):
            s.write(f"First file is not Jan: {stem}\n")
            ok = False
        first_dt = pd.to_datetime(get_ym(files[0]), format="%Y%m")

        # Last is Dec
        stem = files[-1].stem
        if not stem.endswith("12"):
            s.write(f"Last file is not Dec: {stem}\n")
            ok = False
        last_dt = pd.to_datetime(get_ym(files[-1]), format="%Y%m")

        # No gaps
        dt_range = pd.date_range(first_dt, last_dt, freq="MS")
        yms_should_be = [f"{d.year:04d}{d.month:02d}" for d in dt_range]
        yms_are = [get_ym(p) for p in files]
        missing = set(yms_should_be) - set(yms_are)
        if missing:
            s_missing = "\n".join(f"- {m}" for m in sorted(missing))
            s.write(f"Missing months:\n{s_missing}\n")
            ok = False

    if not ok:
        raise AssertionError(s.getvalue())


def track(files):
    """Track a mod/obs period case, saving CE GeoParquet file."""
    case = Case.from_path(files[0])
    print(f"Tracking {case!r}")

    ces = []
    times = []
    for p in files:
        gdf = gpd.read_parquet(p)
        for t, ce in gdf.groupby("time"):
            ces.append(ce)
            times.append(t)

    print(len(times), "times")
    durations = [pd.Timedelta("1h")] * len(times)

    tic = time.perf_counter_ns()
    ce = tams.track(ces, times, durations=durations)

    toc = time.perf_counter_ns()
    dt = pd.Timedelta(toc - tic, unit="ns")
    print(f"Tracking completed in {dt}")

    ce.to_parquet(
        OUT / f"{case.to_id()}.parquet",
        geometry_encoding=GP_ENCODING,
        schema_version=GP_SCHEMA_VERSION,
    )

    return ce


JOB_TPL_TRACK = r"""
#!/bin/bash
#PBS -N mesaclip2
#PBS -q casper
#PBS -l walltime=12:00:00
#PBS -l select=1:ncpus=2:mem=16gb
#PBS -j oe
#PBS -d /glade/u/home/zmoon/git/TAMS/examples/mesaclip

py=/glade/u/home/zmoon/mambaforge/envs/tams/bin/python

$py -c "from mesaclip import Case, get_pre_files, track;
track(get_pre_files()[{case!r}])"
""".lstrip()


def submit_tracks():
    """Submit jobs to track."""
    pre_files = get_pre_files()
    for case, _ in pre_files.items():
        job = JOB_TPL_TRACK.format(case=case)
        stem = f"mesaclip2_{case.to_id(concise=False)}"
        submit_job(job, stem=stem)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(prog="mesaclip")
    subparsers = parser.add_subparsers(dest="command", required=True)

    pre_parser = subparsers.add_parser("pre", help="Submit preprocessing jobs")
    null_tb_parser = subparsers.add_parser("null-tb", help="Submit jobs to find null Tb times")
    check_pre_parser = subparsers.add_parser(
        "check-pre", help="Check that all preprocessed files are present"
    )
    track_parser = subparsers.add_parser("track", help="Submit tracking jobs")

    args = parser.parse_args()

    if args.command == "pre":
        submit_pres()
    elif args.command == "null-tb":
        submit_null_tb()
    elif args.command == "check-pre":
        check_pre_files()
    elif args.command == "track":
        submit_tracks()
