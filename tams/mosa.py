"""
MOSA - MCSs over South America
"""
from __future__ import annotations

import datetime
import subprocess
import warnings
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from typing_extensions import Literal, assert_never

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
    assert ds.isel(time=1).pr.isnull().all() or (ds.pr.isel(time=0) == ds.pr.isel(time=1)).all()
    ds = ds.mean(dim="time", keep_attrs=True)
    assert (ds.lon < 0).any()

    return ds


def preproc_file(fp, *, kind, out_dir=None) -> None:
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
        raise ValueError(f"invalid `which` {kind!r}")
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


def run_wrf_preproced(
    fps: list[Path],
    *,
    id_: str = None,
    rt: Literal["df", "ds", "gdf"] = "df",
    grid: xr.Dataset | None = None,
) -> xr.Dataset | pd.DataFrame:
    """On preprocessed files, do the remaining steps:
    track, classify.

    Note that this returns all tracked CEs, including those not classified as MCS
    (output df includes reason).

    Parameters
    ----------
    id_
        Just used for the info messages, to differentiate when running multiple at same time.
    rt
        Return type.
        - df -- return pandas dataframe (only stats, no contours/geometry)
        - ds -- return xarray dataset (mask to identify MCSs)
            - `mcs_mask` variable, with dims (time, y, x)
            - 'lat'/'lon' with dims (y, x) (even if they only vary in one dim)
            - file name: `<last_name>_WY<YYYY>_<DATA>_SAAG-MCS-mask-file.nc`
        - gdf -- GeoDataFrame, including the contour polygons
    grid
        If using ``rt='ds'``, need a file to get the lat/lon grid from in order to make the masks.
        This assumes that the grid is constant.
    """
    import geopandas as gpd

    import tams

    allowed_rt = {"df", "ds", "gdf"}
    if rt not in allowed_rt:
        raise ValueError(f"`rt` must be one of {allowed_rt}")

    if rt == "ds" and grid is None:
        raise ValueError("`grid` dataset must be provided")

    #
    # Read
    #

    pre = f"[{id_}] " if id_ is not None else ""

    def printt(s):
        """Print message and current time"""
        import datetime

        st = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"{pre}{st}: {s}")

    printt(f"Reading {len(fps)} pre-processed files")
    sts = []  # datetime strings
    dfs = []
    for fp in sorted(fps):
        sts.append(fp.name[12:25])
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

    times = pd.to_datetime(sts, format=r"%Y-%m-%d_%H")

    #
    # Track
    #

    printt("Tracking")
    ce = tams.track(dfs, times)

    #
    # Classify (CEs)
    #

    printt("Classifying")
    ce["mcs_id"] = ce.mcs_id.astype(int)  # TODO: should be already
    n_ = ce.mcs_id.max() + 1
    n = int(n_)
    if n != n_:
        warnings.warn(f"{pre}max MCS ID + 1 was {n_} but using {n}", stacklevel=2)
    is_mcs_list: list[None | bool] = [None] * n
    reason_list: list[None | str] = [None] * n
    for mcs_id, g in ce.groupby("mcs_id"):
        # Compute time
        t = g.time.unique()
        tmin = t.min()
        tmax = t.max()
        duration = pd.Timedelta(tmax - tmin)

        # TODO: collect reasons

        # Assuming instantaneous times, need 5 h for the 4 continuous h criteria
        # but for accumulated (during previous time step), 4 is fine(?)
        n = 4
        if duration < pd.Timedelta(f"{n}H"):
            is_mcs_list[mcs_id] = False
            reason_list[mcs_id] = "duration"
            continue

        # Sum area over cloud elements
        area = g.groupby("itime")["area_km2"].sum()

        # 1. Assess area criterion
        # NOTE: rolling usage assuming data is hourly
        yes = (area >= 40_000).rolling(n, min_periods=0).count().eq(n).any()
        if not yes:
            is_mcs_list[mcs_id] = False
            reason_list[mcs_id] = "area"
            continue

        # Agg min precip over cloud elements
        maxpr = g.groupby("itime")["max_pr"].max()

        # 2. Assess minimum pixel-peak precip criterion
        yes = (maxpr >= 10).rolling(n, min_periods=0).count().eq(n).any()
        if not yes:
            is_mcs_list[mcs_id] = False
            reason_list[mcs_id] = "peak precip"
            continue

        # Compute rainfall volume
        g["prvol"] = g.area_km2 * g.mean_pr  # per CE
        prvol = g.groupby("itime")["prvol"].sum()

        # 3. Assess minimum rainfall volume criterion
        yes = (prvol >= 20_000).sum() >= 1
        if not yes:
            is_mcs_list[mcs_id] = False
            reason_list[mcs_id] = "rainfall volume"
            continue

        # 4. Overshoot threshold currently met for all due to TAMS approach

        # If make it to here, is MCS
        is_mcs_list[mcs_id] = True
        reason_list[mcs_id] = ""

    assert len(is_mcs_list) == len(reason_list) == ce.mcs_id.max() + 1
    assert not any(x is None for x in is_mcs_list)
    assert not any(x is None for x in reason_list)

    ce = ce.drop(columns=["is_mcs"], errors="ignore").merge(
        pd.Series(is_mcs_list, index=range(len(is_mcs_list)), name="is_mcs"),
        how="left",
        left_on="mcs_id",
        right_index=True,
    )
    ce = ce.drop(columns=["not_is_mcs_reason"], errors="ignore").merge(
        pd.Series(reason_list, index=range(len(is_mcs_list)), name="not_is_mcs_reason"),
        how="left",
        left_on="mcs_id",
        right_index=True,
    )

    assert (ce.query("is_mcs == True").not_is_mcs_reason == "").all()
    assert (ce.query("is_mcs == False").not_is_mcs_reason != "").all()

    #
    # Clean up the table
    #

    if rt == "gdf":
        return ce

    printt("Processing CE output")
    cen = ce.geometry.to_crs("EPSG:32663").centroid.to_crs("EPSG:4326")

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", category=UserWarning, message="ellipse model failed for POLYGON"
        )
        eccen = ce.geometry.apply(tams.calc_ellipse_eccen)

    # TODO: include computed stats from above like duration somehow?

    # TODO: separate into a step 3, using the saved gdf?

    if rt == "df":

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
            "is_mcs",
            "not_is_mcs_reason",
        ]

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

        printt("Done")

        return df

    elif rt == "ds":
        import regionmask
        from shapely.errors import ShapelyDeprecationWarning

        time = sorted(ce.time.unique())

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
            x2 = gb[["is_mcs", "not_is_mcs_reason"]].agg(tams.util._the_unique)
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
        assert HERE.parent.name == "TAMS", "repo"
        try:
            cmd = ["git", "-C", HERE.parent.as_posix(), "rev-parse", "--verify", "--short", "HEAD"]
            cp = subprocess.run(cmd, text=True, capture_output=True)
        except Exception:
            ver = ""
        else:
            ver = f" ({cp.stdout.strip()})"
        now = datetime.datetime.utcnow().strftime(r"%Y-%m-%d %H:%M UTC")
        ds.attrs.update(prov=(f"Creating using TAMS{ver} at {now}."))

        # Add the extra variables
        df = pd.concat(dfs, axis="index")
        df[["area_km2", "area_core_km2"]] = df[["area_km2", "area_core_km2"]].astype(np.float64)
        df[["is_mcs"]] = df[["is_mcs"]].astype(np.bool_)
        ds2 = df.reset_index().set_index(["mcs_id", "time"]).to_xarray()
        ds2["mcs_id"] = ds2.mcs_id + 1
        ds = ds.merge(ds2, join="exact", compat="equals")

        # TODO: mcs_id could be uint32, float ones float32, ce_count int32 or uint32 with 0 for null?

        printt("Done")

        return ds

    assert_never(rt)


if __name__ == "__main__":
    # import geopandas as gpd

    import tams

    ds = tams.load_example_mpas().rename(tb="ctt", precip="pr").isel(time=1)

    # # Load and check
    # df2 = gpd.read_parquet("t.parquet")
    # assert df2.equals(df)
