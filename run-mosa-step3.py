"""
Convert from CE gdf output to summary formats (step 3)
"""
import re
from calendar import isleap
from pathlib import Path
from typing import Any, Hashable

import numpy as np
import pandas as pd

from tams.mosa import gdf_to_df, gdf_to_ds

IN_DIR = Path("/glade/scratch/zmoon/mosa2")
OUT_DIR = IN_DIR

re_gdf_fn = re.compile(r"(wrf|gpm)_wy([0-9]{4})\.parquet")

gdf_fps = sorted(IN_DIR.glob("???_wy????.parquet"))


def run_gdf(fp: Path) -> None:
    import geopandas as gpd
    import xarray as xr

    m = re_gdf_fn.fullmatch(fp.name)
    assert m is not None, f"{fp.name!r} should match {re_gdf_fn.pattern}"
    which, s_wy = m.groups()
    wy = int(s_wy)

    if which == "wrf":
        which_mosa = "WRF"
        p_grid = (
            f"/glade/campaign/mmm/c3we/prein/SouthAmerica/MCS-Tracking/"
            f"WY{wy}/WRF/"
            f"tb_rainrate_{wy - 1}-06-01_00:00.nc"
        )
    elif which == "gpm":
        which_mosa = "OBS"
        p_grid = (
            f"/glade/campaign/mmm/c3we/prein/SouthAmerica/MCS-Tracking/"
            f"GPM/{wy - 1}/"
            f"merg_{wy - 1}060100_4km-pixel.nc"
        )
    else:
        raise ValueError(f"Unexpected `which` {which!r}")

    gdf = gpd.read_parquet(fp)

    #
    # df
    #

    df = gdf_to_df(gdf)

    # Only CEs associated to MCS (as defined by MOSA)
    df = gdf
    mcs = df[df.is_mcs].reset_index(drop=True).drop(columns=["is_mcs", "not_is_mcs_reason"])
    mcs

    # Save df
    mcs.to_csv(OUT_DIR / f"{which}_wy{wy}.csv.gz", index=False)

    #
    # ds
    #

    grid = xr.open_dataset(p_grid).squeeze()
    if which == "wrf":
        grid = grid.rename_dims({"rlat": "y", "rlon": "x"})

    ds = gdf_to_ds(gdf, grid=grid)

    # Add null first time if WRF
    t0 = pd.Timestamp(ds.time.values[0])
    if which == "wrf":
        assert t0.hour == 1, "first time (hour 0) should be missing (skipped since tb null)"
        ds0 = ds.isel(time=0).copy()
        for vn in ds0.data_vars:
            ds0[vn] = ds0[vn].where(False, 0)  # FIXME ?
        ds = xr.concat(ds0, ds, dim="time")

    times_should_be = pd.date_range(f"{wy - 1}/06/01", f"{wy}/06/01", freq="H")[:-1]
    nt_missing = 0
    for t_ in times_should_be:
        if t_ not in ds.time.values:
            print(f"warning: {t_} not found in {which_mosa}-WY{wy}")
            nt_missing += 1

    if nt_missing:
        ds = ds.reindex(time=times_should_be, method=None, copy=False, fill_value=0)

    # Check ntimes
    nt_should_be = 8784 if isleap(wy) else 8760
    assert ds.time.size == nt_should_be, f"expected {nt_should_be} times, found {ds.time.size}"
    assert (ds.time.diff("time") == np.timedelta64(1, "h")).all()

    # Save ds
    # <last_name>_WY<YYYY>_<DATA>_SAAG-MCS-mask-file.nc
    # DATA can either be OBS or WRF
    encoding: dict[Hashable, dict[str, Any]] = {"mcs_mask": {"zlib": True, "complevel": 5}}
    ds.to_netcdf(OUT_DIR / f"TAMS_WY{wy}_{which_mosa}_SAAG-MCS-mask-file_all.nc", encoding=encoding)  # type: ignore[arg-type]

    # Drop those not identified as MCSs using the MOSA criteria
    is_mcs = ds.is_mcs.to_series().groupby("mcs_id").agg(lambda x: x[~x.isnull()].unique())
    assert is_mcs.apply(len).eq(1).all()
    is_mcs = is_mcs.explode()
    ids = is_mcs[is_mcs].index
    ds2 = ds.sel(mcs_id=ids)
    assert ds2.is_mcs.all()
    ds2 = ds2.drop_vars(["is_mcs", "not_is_mcs_reason"])
    ds2.to_netcdf(OUT_DIR / f"TAMS_WY{wy}_{which_mosa}_SAAG-MCS-mask-file.nc", encoding=encoding)  # type: ignore[arg-type]


if __name__ == "__main__":
    import joblib

    joblib.Parallel(n_jobs=-2, verbose=1)(joblib.delayed(run_gdf)(fp) for fp in gdf_fps)
