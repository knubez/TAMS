"""
MOSA - MCSs over South America
"""
import warnings
from pathlib import Path

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
    df = tams.data_in_contours(ds.pr, ce, agg=("mean", "max", "min", "count"), merge=True)

    # Save to file
    # Get `pyarrow` from conda-forge
    # GeoParquet spec v0.4.0 requires GeoPandas v0.11 (which no longer warns)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*initial implementation of Parquet.*")
        df.to_parquet(ofp)

    ds.close()


if __name__ == "__main__":
    # import geopandas as gpd

    import tams

    ds = tams.load_example_mpas().rename(tb="ctt", precip="pr").isel(time=1)

    # # Load and check
    # df2 = gpd.read_parquet("t.parquet")
    # assert df2.equals(df)
