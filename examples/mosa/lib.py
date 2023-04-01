"""
MOSA - MCSs over South America

This library defines load and run functions
for the MOSA data
for the different steps of the TAMS workflow.
"""
from __future__ import annotations

import datetime
import subprocess
import warnings
from functools import partial
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr

BASE_DIR = Path("/glade/campaign/mmm/c3we/prein/SouthAmerica/MCS-Tracking")
"""Base location on NCAR GLADE.

Note that this location is only accessibly on Casper, not Hera!

├── GPM
│   ├── 2000
│   ├── 2001
│   ├── ...
│   ├── 2019
│   └── 2020
├── WY2011
│   └── WRF
├── WY2016
│   ├── GPM
│   └── WRF
└── WY2019
    ├── GPM
    └── WRF

Files in the first GPM dir are like `merg_2000081011_4km-pixel.nc`.
WRF files are like `tb_rainrate_2010-11-30_02:00.nc`.
"""

OUT_BASE_DIR = Path("/glade/scratch/knocasio/SAAG")

HERE = Path(__file__).parent

REPO = HERE.parent.parent
assert REPO.is_dir() and REPO.name == "TAMS", "repo"


def load_wrf(files):
    # with dask.config.set(**{'array.slicing.split_large_chunks': False}):
    ds0 = xr.open_mfdataset(files, concat_dim="time", combine="nested", parallel=True)

    ds = (
        ds0.rename({"rainrate": "pr", "tb": "ctt"}).rename_dims({"rlat": "y", "rlon": "x"})
        # .isel(time=slice(1, None))
    )
    ds = ds.assign_coords(lon=(((ds.lon + 180) % 360) - 180))

    return ds


def _load_wrf_file(fp) -> xr.Dataset:
    # fns are like 'tb_rainrate_2015-06-01_12:00.nc'
    # /glade/campaign/mmm/c3we/prein/SouthAmerica/MCS-Tracking/WY2016/WRF/tb_rainrate_2015-06-01_12:00.nc
    ds = (
        xr.open_dataset(fp)
        .rename({"rainrate": "pr", "tb": "ctt"})
        .rename_dims({"rlat": "y", "rlon": "x"})
        .squeeze()
    )
    ds = ds.assign_coords(lon=(((ds.lon + 180) % 360) - 180))

    return ds


def _load_gpm_file(fp) -> xr.Dataset:
    # fns are like 'merg_2015060112_4km-pixel.nc'
    # /glade/campaign/mmm/c3we/prein/SouthAmerica/MCS-Tracking/GPM/2015/merg_2015060112_4km-pixel.nc
    #
    # > The original 30 min IMERG precipitation data have been averaged to hourly
    # > (at the first time stamp), but the original 30 min Tb data are retained
    # > in the files.
    # > Note that the GPM_MERGIR data occasionally have missing Tb data,
    # > either caused by a missing satellite scan, or in lines of individual pixels
    # > that may affect convective cloud object identification.
    ds = (
        xr.open_dataset(fp)
        .drop_vars(["lat_bnds", "lon_bnds", "gw", "area"], errors="ignore")
        .rename({"Tb": "ctt", "precipitationCal": "pr"})
    )
    assert set(ds.data_vars) == {"ctt", "pr"}
    # It seems like most of the files have pr all NaN in the second time,
    # though some have a copy of the first time values,
    # and a few have different values
    # e.g. merg_20110208{15,16,17}_4km-pixel.nc
    # (in which case we don't want to time-average, assuming the above info is correct).
    if not (ds.isel(time=1).pr.isnull().all() or (ds.pr.isel(time=0) == ds.pr.isel(time=1)).all()):
        warnings.warn(
            f"precip at time 1 not all NaN and not all equal to time 0: {fp.name}", stacklevel=3
        )
    ds["pr"] = ds.pr.isel(time=0)
    ds = ds.mean(dim="time", keep_attrs=True)
    assert (ds.lon < 0).any()

    return ds


def load_preproc_one(fp, *, kind: str) -> gpd.GeoDataFrame:
    """Load a preprocessed file (one time) saved in Parquet format."""
    if kind.lower() == "wrf":
        fn_t_fmt = r"%Y-%m-%d_%H"
        fn_t_slice = slice(12, 25)
    elif kind.lower() == "gpm":
        fn_t_fmt = r"%Y%m%d%H"
        fn_t_slice = slice(5, 15)
    else:
        raise ValueError(f"invalid `kind` {kind!r}")

    st = fp.name[fn_t_slice]
    df = gpd.read_parquet(fp)
    # At least when concatting all of them, getting some weird types in WY2016
    # (WY2011 is ok for some reason)
    # TODO: alleviate need for this (in preproc)
    df = df.assign(
        mean_pr=df.mean_pr.astype(float),
        max_pr=df.max_pr.astype(float),
        min_pr=df.min_pr.astype(float),
        count_pr=df.count_pr.astype(int),
    ).convert_dtypes()

    time = pd.to_datetime(st, format=fn_t_fmt)
    df.attrs.update(time=time)

    return df


def load_preproc_zip(fp, *, kind: str) -> gpd.GeoDataFrame:
    """Load zip of preprocessed identify files in Parquet format."""
    import zipfile

    with zipfile.ZipFile(fp) as zf:
        files = zf.namelist()
        dfs = []
        for fn in files:
            with zf.open(fn) as f:
                dfs.append(load_preproc_one(f, kind=kind))

    dfs.sort(key=lambda df: df.attrs["time"])

    times = [df.attrs["time"] for df in dfs]

    return times, dfs


def preproc_file(fp, *, kind: str, out_dir=None) -> None:
    """Pre-process file, saving CE dataset, including CE precip stats, to file."""
    import tams

    fp = Path(fp)
    ofn = f"{fp.stem}_ce.parquet"
    if out_dir is None:
        ofp = OUT_BASE_DIR / "pre" / ofn
    else:
        ofp = Path(out_dir) / ofn

    if kind.lower() == "wrf":
        ds = _load_wrf_file(fp)
    elif kind.lower() == "gpm":
        ds = _load_gpm_file(fp)
    else:
        raise ValueError(f"invalid `kind` {kind!r}")
    assert len(ds.dims) == 2

    # Identify CEs
    ce, _ = tams.core._identify_one(ds.ctt, ctt_threshold=241, ctt_core_threshold=225)
    ce = (
        ce[["geometry", "area_km2", "area219_km2"]]
        .rename(columns={"area219_km2": "area_core_km2"})
        .convert_dtypes()
    )

    # Get precip stats
    agg = ("mean", "max", "min", "count")
    try:
        df = tams.data_in_contours(ds.pr, ce, agg=agg, merge=True)
    except ValueError as e:
        if str(e) == "no data found in contours":
            print(f"warning: no pr data in contours for {fp.name}")
            df = ce
            for a in agg:
                df[f"{a}_pr"] = np.nan
        else:
            raise

    # Save to file
    # Get `pyarrow` from conda-forge
    # GeoParquet spec v0.4.0 requires GeoPandas v0.11 (which no longer warns)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*initial implementation of Parquet.*")
        df.to_parquet(ofp)
        # TODO: avoid `:` in fn since Windows doesn't like?

    ds.close()


preproc_wrf_file = partial(preproc_file, kind="wrf")
preproc_gpm_file = partial(preproc_file, kind="gpm")


_classify_cols = [
    "is_mcs",
    "meets_crit_duration",
    "meets_crit_area",
    "meets_crit_prpeak",
    "meets_crit_prvol",
]
_classify_cols_set = set(_classify_cols)
_classify_stats_cols = [
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
    assert res.keys() == _classify_cols_set

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

    return pd.Series(res)


def classify(
    ce: gpd.GeoDataFrame,
    *,
    pre: str = "",
    include_stats: bool = False,
) -> gpd.GeoDataFrame:
    """Determine if CE groups (MCS IDs) is indeed MCS or not under the MOSA criteria.

    Modifies `ce` in-place.

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

    mcs_info = ce.groupby("mcs_id").apply(classify_one, include_stats=include_stats, pre=pre)
    assert mcs_info.index.name == "mcs_id"

    ce = ce.drop(columns=_classify_cols + _classify_stats_cols, errors="ignore").merge(
        mcs_info, how="left", left_on="mcs_id", right_index=True
    )

    return ce


def run_preproced(
    fps: list[Path],
    *,
    kind: str,
    id_: str | None = None,
    track_kws: dict[str, Any] | None = None,
) -> gpd.GeoDataFrame:
    """On preprocessed files, do the remaining steps:
    track, classify.

    Note that this returns all tracked CEs, including those not classified as MCS
    (output gdf includes reason).

    Returns GeoDataFrame including the contour polygons.

    Parameters
    ----------
    id_
        Just used for the info messages, to differentiate when running multiple at same time.
    """
    import geopandas as gpd

    import tams

    #
    # Read
    #

    pre = f"[{id_}] " if id_ is not None else ""

    def printt(s):
        """Print message and current time"""
        import datetime

        st = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"{pre}{st}: {s}")

    # TODO: replace with zip load?
    if kind.lower() == "wrf":
        fn_t_fmt = r"%Y-%m-%d_%H"
        fn_t_slice = slice(12, 25)
    elif kind.lower() == "gpm":
        fn_t_fmt = r"%Y%m%d%H"
        fn_t_slice = slice(5, 15)
    else:
        raise ValueError(f"invalid `kind` {kind!r}")

    printt(f"Reading {len(fps)} pre-processed files")
    sts = []  # datetime strings
    dfs = []
    for fp in sorted(fps):
        sts.append(fp.name[fn_t_slice])
        df = gpd.read_parquet(fp)
        # At least when concatting all of them, getting some weird types in WY2016
        # (WY2011 is ok for some reason)
        # TODO: alleviate need for this (in preproc)
        df = df.assign(
            mean_pr=df.mean_pr.astype(float),
            max_pr=df.max_pr.astype(float),
            min_pr=df.min_pr.astype(float),
            count_pr=df.count_pr.astype(int),
        ).convert_dtypes()
        dfs.append(df)

    times = pd.to_datetime(sts, format=fn_t_fmt)

    #
    # Track
    #

    if track_kws is None:
        track_kws = {}

    printt("Tracking")
    ce = tams.track(dfs, times, **track_kws)

    #
    # Classify (CEs)
    #

    printt("Classifying")
    ce = classify(ce, pre=pre)

    printt("Done")

    return ce


run_wrf_preproced = partial(run_preproced, kind="wrf")
run_gpm_preproced = partial(run_preproced, kind="gpm")


def gdf_to_df(ce) -> pd.DataFrame:
    """Convert CE gdf to a df of only stats (no contours/geometry)."""
    import tams

    cen = ce.geometry.to_crs("EPSG:32663").centroid.to_crs("EPSG:4326")

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", category=UserWarning, message="ellipse model failed for POLYGON"
        )
        eccen = ce.geometry.apply(tams.calc_ellipse_eccen)

    col_order = [
        "time",
        "lat",
        "lon",
        "area_km2",
        "area_core_km2",
        "eccen",
        "mcs_id",
        "mean_pr",
        "max_pr",
        "min_pr",
        "count_pr",
    ] + _classify_cols

    ce_ = (
        ce.drop(
            columns=[
                # "inds219", "area219_km2", "cs219",
                "itime",
                "dtime",
                "geometry",
            ]
        )
        .assign(eccen=eccen)
        .assign(lat=cen.y, lon=cen.x)
    )

    assert set(ce_.columns) == set(col_order)

    df = pd.DataFrame(ce_)[col_order]
    df

    return df


def gdf_to_ds(ce, *, grid: xr.Dataset) -> xr.Dataset:
    """Convert CE gdf to a MCS mask dataset.

    - `mcs_mask` variable, with dims (time, y, x)
    - 'lat'/'lon' with dims (y, x) (even if they only vary in one dim)
    - file name: `<last_name>_WY<YYYY>_<DATA>_SAAG-MCS-mask-file.nc`

    Parameters
    ----------
    grid
        File to get the lat/lon grid from in order to make the masks.
        (This assumes that the grid is constant.)
    """
    import regionmask
    from shapely.errors import ShapelyDeprecationWarning

    import tams

    time = sorted(ce.time.unique())

    unique_cols = _classify_cols
    if "mcs_id_orig" in ce.columns:
        unique_cols += ["mcs_id_orig"]

    dfs = []
    masks = []
    for t in time:  # TODO: joblib?
        ce_t = ce.loc[ce.time == t]
        # NOTE: `numbers` can't have duplicates, so we use `.dissolve()` to combine
        regions = regionmask.from_geopandas(
            ce_t[["geometry", "mcs_id"]].dissolve(by="mcs_id").reset_index(),
            numbers="mcs_id",
        )
        with warnings.catch_warnings():
            # ShapelyDeprecationWarning: __len__ for multi-part geometries is deprecated and will be removed in Shapely 2.0. Check the length of the `geoms` property instead to get the  number of parts of a multi-part geometry.
            warnings.filterwarnings(
                "ignore",
                category=ShapelyDeprecationWarning,
                message="__len__ for multi-part geometries is deprecated",
            )
            mask = regions.mask(grid)
        masks.append(mask)

        # Agg some other info
        gb = ce_t.groupby("mcs_id")
        x1 = gb[["area_km2", "area_core_km2"]].sum()
        x2 = gb[unique_cols].agg(tams.util._the_unique)
        x3 = gb[["area_km2"]].count().rename(columns={"area_km2": "ce_count"})
        x3["time"] = t

        dfs.append(pd.concat([x1, x2, x3], axis="columns"))

    da = xr.concat(masks, dim="time")
    da["time"] = time

    # Our IDs start at 0, but we want to use 0 for missing/null value
    # (the sample file looks to be doing this)
    assert da.min() == 0
    da = (da + 1).fillna(0).astype(np.int64)
    da.attrs.update(long_name="MCS ID mask", description="Value 0 indicates null (no MCS).")

    ds = da.to_dataset().rename_vars(mask="mcs_mask")

    # Remove current irrelevant 'coordinates' attrs from das
    # ('mcs_mask' will still get `mcs_mask:coordinates = "lon lat"` in the saved file)
    for _, v in ds.variables.items():
        if "coordinates" in v.attrs:
            del v.attrs["coordinates"]

    # Add some info
    try:
        cmd = ["git", "-C", REPO.as_posix(), "rev-parse", "--verify", "--short", "HEAD"]
        cp = subprocess.run(cmd, text=True, capture_output=True)
    except Exception:
        ver = ""
    else:
        ver = f" ({cp.stdout.strip()})"
    now = datetime.datetime.utcnow().strftime(r"%Y-%m-%d %H:%M UTC")
    ds.attrs.update(prov=(f"Created using TAMS{ver} at {now}."))

    # Add the extra variables
    df = pd.concat(dfs, axis="index")
    df[["area_km2", "area_core_km2"]] = df[["area_km2", "area_core_km2"]].astype(np.float64)
    df[["is_mcs"]] = df[["is_mcs"]].astype(np.bool_)
    ds2 = df.reset_index().set_index(["mcs_id", "time"]).to_xarray()
    ds2["mcs_id"] = ds2.mcs_id + 1
    ds = ds.merge(ds2, join="exact", compat="equals")

    # TODO: mcs_id could be uint32, float ones float32, ce_count int32 or uint32 with 0 for null?

    return ds


if __name__ == "__main__":
    # import geopandas as gpd

    import tams

    ds = tams.load_example_mpas().rename(tb="ctt", precip="pr").isel(time=1)

    # # Load and check
    # df2 = gpd.read_parquet("t.parquet")
    # assert df2.equals(df)
