"""
DYAMOND -- DYnamics of the Atmospheric general circulation Modeled On Non-hydrostatic Domains

https://www.esiwace.eu/the-project/past-phases/dyamond-initiative
"""
from __future__ import annotations

import datetime
import re
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

BASE_DIR_IN = Path("/glade/campaign/mmm/c3we/prein/Papers/2023_Zhe-MCSMIP")
"""Base input data directory (Andy's).

Under here we have 'Summer' and 'Winter' dirs (title case)
and then model dirs with 'olr_pcp_instantaneous' subdirs
that have the nc files, one for each hour
e.g. ::

/glade/campaign/mmm/c3we/prein/Papers/2023_Zhe-MCSMIP/Winter/UM/olr_pcp_instantaneous/pr_rlut_um_winter_2020012007.nc
"""

BASE_DIR_OUT = Path("/glade/scratch/zmoon/dyamond")
BASE_DIR_OUT_PRE = BASE_DIR_OUT / "pre"


def _make_vn_map():
    def _to_map(s):
        lines = s.splitlines()
        n = len(lines)
        assert n % 3 == 0
        m = {}
        for i in range(3, n, 3):
            model, pr, olr = lines[i : i + 3]
            if model == "ARPEGE-NH":
                model = "ARPEGE"
            m[model] = {pr: "pr", olr: "olr"}
        return m

    d = {}

    # NOTE: these text blocks were copied and pasted from the Google Doc
    # NOTE: summer and winter renamings aren't necessarily the same

    # Summer
    s = """\
Model
Precipitation
OLR
ARPEGE-NH
param8.1.0
ttr
FV3
pr
flut
IFS
tp
ttr
MPAS
pr
olrtoa
NICAM
sa_tppn
sa_lwu_toa
SAM
Precac
LWNTA
UM
precipitation_flux
toa_outgoing_longwave_flux"""
    d["summer"] = _to_map(s)

    # Winter
    s = """\
Model
Precipitation
OLR
ARPEGE-NH
pr
rlt
GEOS
pr
rlut
GRIST
pr
rlt
ICON
pr
rlut
IFS
pracc
rltacc
MPAS
pr
rltacc
NICAM
sa_tppn
sa_lwu_toa
SAM
pracc
rltacc
SCREAM
pr
rlt
UM
pr
rlut
XSHiELD
pr
rlut"""
    d["winter"] = _to_map(s)

    for season in d:
        d[season]["OBS"] = {"precipitationCal": "pr", "Tb": "tb"}

    return d


VN_MAP = _make_vn_map()
"""Map from season and model to variable rename mapping.

For example::

    d["Summer"]["OBS"] = {"precipitationCal": "pr", "Tb": "tb"}
"""

START = {
    "summer": pd.Timestamp("2016-08-01"),
    "winter": pd.Timestamp("2020-01-20"),
}

SEASONS = ["summer", "winter"]
"""The seasons in lowercase, like they should be."""

assert set(SEASONS) == set(VN_MAP) == set(START)


def get_t_file(fn: str | Path) -> pd.Timestamp:
    """Get time stamp from file name."""
    if isinstance(fn, Path):
        fn = fn.stem
    m = re.search(r"[0-9]{10}", fn)
    if m is None:
        raise ValueError(f"YYYYMMDDHH time not found in '{fn}'")
    return pd.to_datetime(m.group(), format=r"%Y%m%d%H")


def inspect_input_data():
    """Print info about the input datasets,
    checking what files are available and checking the contents of the first one.
    """

    for season in ["Summer", "Winter"]:
        print(season)
        d = BASE_DIR_IN / season
        assert d.is_dir()

        start = (
            datetime.datetime(2016, 8, 1) if season == "Summer" else datetime.datetime(2020, 1, 20)
        )
        time_should_be = [start]
        for _ in range(40 * 24 - 1):
            time_should_be.append(time_should_be[-1] + datetime.timedelta(hours=1))
        assert time_should_be[-1] - time_should_be[0] == datetime.timedelta(days=39, hours=23)

        model_dirs = sorted(d.glob("*"))
        for model_dir in model_dirs:
            model = model_dir.name
            print(" ", model)
            (subdir,) = list(model_dir.glob("*"))  # just one
            assert subdir.name in {"olr_pcp", "olr_pcp_instantaneous"}

            files = sorted(subdir.glob("*"))
            # They don't all have the same number of files
            # but the most common number is 960 (40 days)
            pad = " " * 3
            print(pad, "n =", len(files))

            fp = files[0]
            fn = fp.name
            print(pad, "first:", fn)
            ds = xr.open_dataset(fp)
            if model == "MPAS":
                ds = ds.rename(xtime="time")
            if model == "OBS" and season == "Summer":
                assert ds.dims["time"] == 2
                ds = ds.isel(time=0).expand_dims("time", axis=0)
            assert {"time", "lon", "lat"} <= set(ds.dims)  # some also have 'bnds'
            assert ds.dims["time"] == 1
            print(pad, "t data:", ds.time.values[0])

            t_file = get_t_file(fp.stem)
            print(pad, "t path:", t_file)

            assert ds.dims["lon"] == 3600, "0.1 deg"
            lona, lonb = -179.95, 179.95
            if model == "OBS" and season.lower() == "summer":
                lata, latb = -89.95, 89.95
                nlat = 1800
            else:
                lata, latb = -59.95, 59.95
                nlat = 1200
            assert ds.dims["lat"] == nlat
            lon, lat = ds["lon"], ds["lat"]
            assert lat.dims == ("lat",)
            try:
                assert lat.values[0] == lata and lat.values[-1] == latb
            except AssertionError:
                print(pad, "unexpected lat range:", lat.values[0], "...", lat.values[-1])
            assert lon.dims == ("lon",)
            assert lon.values[0] == lona and np.isclose(lon.values[-1], lonb)

            dvs = sorted(ds.data_vars)
            print(pad, "dvs:", dvs)

            rn = VN_MAP[season.lower()][model]
            assert rn.keys() <= set(dvs), "remap"
            print(pad, "rename:", ", ".join(f"{k!r}=>{v!r}" for k, v in rn.items()))

            # Check which times we have
            t_files = []
            for fp in files:
                ymdh = re.search(r"[0-9]{10}", fp.stem).group()
                t_file = datetime.datetime.strptime(ymdh, r"%Y%m%d%H")
                t_files.append(t_file)
            t_miss = [t for t in time_should_be if t not in t_files]
            if t_miss:
                print(pad, f"missing files for these {len(t_miss)} times:")
                for t in t_miss:
                    print(pad, "-", t.strftime(r"%Y-%m-%d %H"))


# TODO: idealized cases data ('idealized_cases')


def iter_input_paths():
    """Yield paths to input files."""

    for season in SEASONS:
        season_dir = BASE_DIR_IN / season.title()
        assert season_dir.is_dir()

        start = START[season]

        # Some models don't have first hour or first day
        # Zhe said skipping first day is ok
        time_range = pd.date_range(
            start=start + pd.Timedelta("1D"),
            periods=(40 - 1) * 24,
            freq="1H",
        )

        for model in VN_MAP[season]:
            model_dir = season_dir / model
            if not model_dir.is_dir():
                print(f"missing directory {season} | {model}")
                continue

            files = sorted(model_dir.glob("**/*.nc"))
            t_to_file = {get_t_file(fp): fp for fp in files}

            for t in time_range:
                fp = t_to_file.get(t)
                if fp is None:
                    print(f"missing file for {season} | {model} | {t:%Y-%m-%d %H}")
                    continue
                yield fp


def open_input(p: Path) -> xr.Dataset:
    """Open a single input nc file as an `xarray.Dataset`.

    The dataset is prepared for pre-processing.
    It has one time step and 'pr' and 'tb' variables.
    """
    p = p.absolute()

    # Guess season and model from path
    model = p.parent.parent.name
    season = p.parent.parent.parent.name.lower()
    assert season in SEASONS
    assert model in VN_MAP[season]

    t_file = get_t_file(p.stem)

    ds = xr.open_dataset(p)

    # Specific adjustments
    if model == "MPAS":
        ds = ds.rename(xtime="time")
    if model == "OBS" and season == "Summer":
        assert ds.dims["time"] == 2
        ds = ds.isel(time=slice(0, 1))

    # Zhe said to ignore time difference from the hour,
    # just assume it is on the hour like the obs
    ds["time"] = [t_file]

    # Normalize variable names
    rn = VN_MAP[season][model]
    ds = ds.rename_vars(rn)

    # Select variables
    ds = ds[list(rn.values())]

    # For models, compute Tb from OLR
    if model != "OBS":
        from scipy.constants import Stefan_Boltzmann as sigma

        assert ds.data_vars.keys() == {"pr", "olr"}

        # Yang and Sligo (2001)
        # Given by Zhe
        a = 1.228
        b = -1.106e-3
        tf = (ds["olr"] / sigma) ** 0.25
        ds["tb"] = (-a + np.sqrt(a**2 + 4 * b * tf)) / (2 * b)
        ds["tb"].attrs.update(
            long_name="brightness temperature",
            units="K",
        )
        ds = ds.drop_vars("olr")
    assert ds.data_vars.keys() == {"pr", "tb"}

    # Meta
    ds.attrs.update(
        _season=season,
        _model=model,
        _time=t_file.strftime(r"%Y-%m-%d %H"),
    )

    return ds.squeeze()


def preproc_file(p: Path, *, overwrite: bool = True) -> None:
    """Preprocess a single input data nc file and save it to Parquet.

    If `overwrite` is false and the output file already exists,
    pre-processing is skipped.

    NOTE: `pyarrow` is required in order to save the GeoDataFrames to Parquet.
    NOTE: GeoPandas currently only supports saving to disk, not bytes.
    """
    import tams

    ds = open_input(p)
    id_ = f"{ds._season}__{ds._model.lower()}__{ds._time.replace(' ', '_')}"

    p_out = BASE_DIR_OUT_PRE / f"{id_}.parquet"
    if not overwrite and p_out.is_file():
        print(f"skipping {id_} ({p.as_posix()}) (exists)")
        return

    # Identify CEs
    ce, _ = tams.core._identify_one(ds["tb"], ctt_threshold=241, ctt_core_threshold=225)
    ce = ce[["geometry", "area_km2", "area219_km2"]].rename(
        columns={"area219_km2": "area_core_km2"}
    )

    # Get precip stats
    agg = ("mean", "max", "min", "count")
    try:
        df = tams.data_in_contours(ds["pr"], ce, agg=agg, merge=True)
    except ValueError as e:
        if str(e) == "no data found in contours":
            print(f"warning: no pr data in contours for {id_} ({p.as_posix()})")
            df = ce
            for a in agg:
                df[f"{a}_pr"] = np.nan
        else:
            raise

    # Save to Parquet
    df.to_parquet(p_out, schema_version="0.4.0")
