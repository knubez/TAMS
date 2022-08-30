"""
MOSA - MCSs over South America
"""
import warnings
from pathlib import Path

import numpy as np
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


def load_wrf(files):
    # with dask.config.set(**{'array.slicing.split_large_chunks': False}):
    ds0 = xr.open_mfdataset(files, concat_dim="time", combine="nested", parallel=True)

    ds = (
        ds0.rename({"rainrate": "pr", "tb": "ctt"}).rename_dims({"rlat": "y", "rlon": "x"})
        # .isel(time=slice(1, None))
    )
    ds = ds.assign_coords(lon=(((ds.lon + 180) % 360) - 180))

    return ds


def preproc_wrf_file(fp, *, out_dir=None):
    """Pre-process file, saving CE dataset including CE precip stats to file."""
    import tams

    fp = Path(fp)
    ofn = f"{fp.stem}_ce.parquet"
    if out_dir is None:
        ofp = OUT_BASE_DIR / "pre" / ofn
    else:
        ofp = Path(out_dir) / ofn

    ds = (
        xr.open_dataset(fp)
        .rename({"rainrate": "pr", "tb": "ctt"})
        .rename_dims({"rlat": "y", "rlon": "x"})
        .squeeze()
    )
    ds = ds.assign_coords(lon=(((ds.lon + 180) % 360) - 180))
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


def run_wrf_preproced(fps: list[Path]):
    """On preprocessed files, do the remaining steps:
    track, classify.
    """
    import time

    import geopandas as gpd
    import pandas as pd

    import tams

    tic = time.perf_counter()

    #
    # Read
    #

    def printt(s):
        """Print message and current time"""
        import datetime

        st = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        toc = time.perf_counter()
        print(f"{st}: {s} (elapsed: {toc - tic:.3g} s)")

    printt("Reading pre-processed files")
    sts = []  # datetime strings
    dfs = []
    for fp in sorted(fps):
        sts.append(fp.name[12:25])
        dfs.append(gpd.read_parquet(fp))

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
    n = ce.mcs_id.max() + 1
    is_mcs_list = [None] * n
    reason_list = [None] * n
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

    printt("Cleaning up the table")
    cen = ce.geometry.to_crs("EPSG:32663").centroid.to_crs("EPSG:4326")

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", category=UserWarning, message="ellipse model failed for POLYGON"
        )
        eccen = ce.geometry.apply(tams.calc_ellipse_eccen)

    # TODO: include computed stats from above like duration somehow?

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


if __name__ == "__main__":
    # import geopandas as gpd

    import tams

    ds = tams.load_example_mpas().rename(tb="ctt", precip="pr").isel(time=1)

    # # Load and check
    # df2 = gpd.read_parquet("t.parquet")
    # assert df2.equals(df)
