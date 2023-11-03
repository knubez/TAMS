"""
DYAMOND -- DYnamics of the Atmospheric general circulation Modeled On Non-hydrostatic Domains

https://www.esiwace.eu/the-project/past-phases/dyamond-initiative
"""
from __future__ import annotations

import datetime
import re
import warnings
from pathlib import Path
from typing import Any

import geopandas as gpd
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

    d["summer"]["OBS"] = {"precipitationCal": "pr", "Tb": "tb"}
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


def get_preproced_paths() -> dict[tuple[str, str], list[Path]]:
    """Return dict of `(season, model)` to list of file paths."""
    from collections import defaultdict

    paths = defaultdict(list)
    for p in BASE_DIR_OUT_PRE.glob("*.parquet"):
        season, model, _ = p.stem.split("__")
        paths[(season, model)].append(p)

    for key in paths:
        paths[key].sort()

    return paths


_CLASSIFY_COLS = [
    "is_mcs",
    "meets_crit_duration",
    "meets_crit_area",
    "meets_crit_prpeak",
    "meets_crit_prvol",
]
_CLASSIFY_COLS_SET = set(_CLASSIFY_COLS)
_CLASSIFY_STATS_COLS = [
    "duration",
    "max_area",
    "max_maxpr",
    "max_prvol",
]


def classify_one(g: gpd.GeoDataFrame, *, pre: str = "", include_stats: bool = False) -> pd.Series:
    """Determine if CE group (MCS ID) is indeed MCS or not under the MOSA criteria.

    Returns Series of stats for the CE group (MCS ID).

    Parameters
    ----------
    g
        CE dataset (output from TAMS tracking step, single MCS ID).
    pre
        Prefix added to warning/info messages to identify a specific run.
    """
    if "mcs_id" in g:
        assert g.mcs_id.nunique() == 1

    # Compute duration
    t = g.time.unique()
    tmin = t.min()
    tmax = t.max()
    duration = pd.Timedelta(tmax - tmin)

    # Assuming instantaneous times, need 5 h for the 4 continuous h criteria
    # but for accumulated (during previous time step), 4 is fine(?) (according to Andy)
    n = 4
    meets_crit_duration = duration >= pd.Timedelta(f"{n}H")
    # TODO: ^ not really one of the 4 criteria (though needed for 1 and 2)

    # Compute rainfall volume
    ce_prvol = g.area_km2 * g.mean_pr  # per CE

    # Group by time
    gb = g.assign(prvol=ce_prvol).groupby("itime")

    # Sum area over cloud elements
    area = gb["area_km2"].sum()

    # Agg max precip over cloud elements
    maxpr = gb["max_pr"].max()

    # Sum rainfall volume over cloud elements
    prvol = gb["prvol"].sum()

    # 1. Assess area criterion
    # NOTE: rolling usage assuming data is hourly
    meets_crit_area = (area >= 40_000).rolling(n, min_periods=0).sum().eq(n).any()

    # 2. Assess minimum pixel-peak precip criterion
    meets_crit_prpeak = (maxpr >= 10).rolling(n, min_periods=0).sum().eq(n).any()

    # 3. Assess minimum rainfall volume criterion
    meets_crit_prvol = (prvol >= 20_000).sum() >= 1

    # 4. Overshoot threshold currently met for all due to TAMS approach
    # TODO: check anyway?

    # An MCS meets all of the criteria
    is_mcs = all(
        [
            meets_crit_duration,
            meets_crit_area,
            meets_crit_prpeak,
            meets_crit_prvol,
        ]
    )

    res = {
        "is_mcs": is_mcs,
        "meets_crit_duration": meets_crit_duration,
        "meets_crit_area": meets_crit_area,
        "meets_crit_prpeak": meets_crit_prpeak,
        "meets_crit_prvol": meets_crit_prvol,
    }
    assert res.keys() == _CLASSIFY_COLS_SET

    # Sanity checks
    max_area = area.max()
    if meets_crit_area:
        assert max_area >= 40_000
    max_maxpr = maxpr.max()
    if meets_crit_prpeak:
        assert max_maxpr >= 10
    max_prvol = prvol.max()
    if meets_crit_prvol:
        assert max_prvol >= 20_000

    if include_stats:
        res.update(
            {
                "duration": duration,
                "max_area": max_area,
                "max_maxpr": max_maxpr,
                "max_prvol": max_prvol,
            }
        )
        assert res.keys() > set(_CLASSIFY_STATS_COLS)

    return pd.Series(res)


def classify(
    ce: gpd.GeoDataFrame,
    *,
    pre: str = "",
    include_stats: bool = False,
) -> gpd.GeoDataFrame:
    """Determine if CE groups (MCS IDs) are indeed MCS or not under the MOSA criteria.

    Parameters
    ----------
    ce
        CE dataset (output from TAMS tracking step).
    pre
        Prefix added to warning/info messages to identify a specific run.
    """
    ce["mcs_id"] = ce.mcs_id.astype(int)
    # TODO: ^ should be already (unless NaNs due to days with no CEs identified)
    n_mcs_ = ce.mcs_id.max() + 1
    n_mcs = int(n_mcs_)
    if n_mcs != n_mcs_:
        warnings.warn(f"{pre}max MCS ID + 1 was {n_mcs_} but using {n_mcs}", stacklevel=2)

    # TODO: parallel option?
    mcs_info = ce.groupby("mcs_id").apply(classify_one, include_stats=include_stats, pre=pre)
    assert mcs_info.index.name == "mcs_id"

    ce = ce.drop(columns=_CLASSIFY_COLS + _CLASSIFY_STATS_COLS, errors="ignore").merge(
        mcs_info,
        how="left",
        left_on="mcs_id",
        right_index=True,
    )

    return ce


def run(
    fps: list[Path],
    *,
    id_: str | None = None,
    track_kws: dict[str, Any] | None = None,
) -> gpd.GeoDataFrame:
    """On preprocessed files, do the remaining steps:
    track, classify.

    Note that this returns all tracked CEs, including those not classified as MCS
    (the output gdf includes reason).

    Returns GeoDataFrame including the contour polygons.

    Parameters
    ----------
    fps
        GeoDataFrames saved in Parquet format,
        each corresponding to the CEs identified at a single time step.
    id_
        Just used for the info messages, to differentiate when running multiple at same time.
    track_kws
        Passed to :func:`tams.track`.

    See Also
    --------
    classify
    classify_one
    tams.track
    """
    import tams

    #
    # Read
    #

    pre = f"[{id_}] " if id_ is not None else ""

    def printt(s):
        """Print message and current time"""
        import datetime

        st = datetime.datetime.now().strftime(r"%Y-%m-%d %H:%M:%S")
        print(f"{pre}{st}: {s}")

    printt(f"Reading {len(fps)} pre-processed files")
    sts = []  # datetime strings
    ces = []
    for fp in sorted(fps):
        sts.append(fp.stem.split("__")[-1])
        df = gpd.read_parquet(fp)
        ces.append(df)

    times = pd.to_datetime(sts, format=r"%Y-%m-%d_%H")

    #
    # Track
    #

    if track_kws is None:
        track_kws = {}

    printt("Tracking")
    ce = tams.track(ces, times, **track_kws)

    #
    # Classify (CEs)
    #

    printt("Classifying")
    ce = classify(ce, pre=pre)

    printt("Done")

    return ce
