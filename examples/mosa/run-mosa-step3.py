"""
Convert from CE gdf output to summary formats (step 3)
"""
import re
from calendar import isleap
from pathlib import Path
from typing import Any, Hashable

import numpy as np
import pandas as pd

from lib import _classify_cols, gdf_to_df, gdf_to_ds, re_id

IN_DIR = Path("/glade/derecho/scratch/zmoon/mosa")
OUT_DIR = IN_DIR

do_bench = True

re_gdf_fn = re.compile(r"(wrf|gpm)_wy([0-9]{4})\.parquet")

gdf_fps = sorted(IN_DIR.glob("???_wy????.parquet"))

ds_null_val = {
    "mcs_mask": 0,
    "area_km2": np.nan,
    "area_core_km2": np.nan,
    "ce_count": np.nan,
}
for col in _classify_cols:
    ds_null_val[col] = np.nan


def run_gdf(fp: Path, *, bench: bool = False) -> None:
    import geopandas as gpd
    import xarray as xr

    if bench:
        which = "wrf"
        wy = 2019
    else:
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

    if bench:
        out_stem_simple = "bench"
        out_name_mosa = "bench.nc"
    else:
        out_stem_simple = f"{which}_wy{wy}"
        out_name_mosa = f"TAMS_WY{wy}_{which_mosa}_SAAG-MCS-mask-file.nc"

    # Load CE gdf, which has non-MCS CEs, but classified to indicate so
    gdf = gpd.read_parquet(fp)

    #
    # df
    #

    df = gdf_to_df(gdf)

    # Only CEs associated to MCS (as defined by MOSA)
    df_mcs = df[df.is_mcs].reset_index(drop=True).drop(columns=_classify_cols)
    df_mcs

    # Save df
    df_mcs.to_csv(OUT_DIR / f"{out_stem_simple}.csv.gz", index=False)

    #
    # ds
    #

    grid = xr.open_dataset(p_grid).squeeze()
    if which == "wrf":
        grid = grid.rename_dims({"rlat": "y", "rlon": "x"})
    grid.lon.attrs.update(units="degree_east")

    # Drop non-MCSs
    gdf_mcs = gdf[gdf.is_mcs].drop(columns=_classify_cols)
    assert gdf_mcs.mcs_id.nunique() < gdf.mcs_id.nunique()
    gdf_mcs_reid = gdf_mcs.assign(mcs_id=re_id(gdf_mcs), mcs_id_orig=gdf_mcs.mcs_id)

    # Create ds, featuring (time, y, x) mask array
    ds = gdf_to_ds(gdf_mcs_reid, grid=grid)

    # For bench case, just write out current result
    if bench:
        encoding: dict[Hashable, dict[str, Any]] = {"mcs_mask": {"zlib": True, "complevel": 1}}
        ds.to_netcdf(OUT_DIR / out_name_mosa, encoding=encoding)  # type: ignore[arg-type]
        return

    # Add null first time if WRF
    t0 = pd.Timestamp(ds.time.values[0])
    t0_should_be = pd.Timestamp(f"{wy - 1}/06/01")
    if which == "wrf":
        assert t0 != t0_should_be, "first time (hour 0) should be missing (skipped since tb null)"
        ds0 = ds.isel(time=0).copy()
        ds0["time"] = t0_should_be
        assert ds0.time.dt.hour == 0
        vns_non_time = [vn for vn in ds.data_vars if "time" not in ds[vn].dims]
        for vn in ds0.data_vars:
            assert isinstance(vn, str)
            if vn in vns_non_time:
                continue
            if vn not in ds_null_val:
                raise Exception(
                    f"null value not known for data var {vn}. "
                    f"Known for: {sorted(ds_null_val)}. "
                    f"Data vars: {sorted(ds.data_vars)}."
                )
            ds0[vn] = ds0[vn].where(False, ds_null_val[vn])
        ds = xr.concat([ds0.drop_vars(vns_non_time), ds.drop_vars(vns_non_time)], dim="time")
        for vn in vns_non_time:
            ds[vn] = ds0[vn]

    times_should_be = pd.date_range(f"{wy - 1}/06/01", f"{wy}/06/01", freq="H")[:-1]
    nt_missing = 0
    for t_ in times_should_be:
        if t_ not in ds.time.values:
            print(f"warning: {t_} not found in {which_mosa}-WY{wy}")
            nt_missing += 1

    if nt_missing:
        ds = ds.reindex(time=times_should_be, method=None, copy=False, fill_value=ds_null_val)

    # Check ntimes
    nt_should_be = 8784 if isleap(wy) else 8760
    assert ds.time.size == nt_should_be, f"expected {nt_should_be} times, found {ds.time.size}"
    assert (ds.time.diff("time") == np.timedelta64(1, "h")).all()

    # Check mask IDs consistent with dim coord
    mcs_mask_unique = np.unique(ds.mcs_mask)
    assert set(mcs_mask_unique) - set(ds.mcs_id.values) == {0}

    # Check for consistency with non-MCS gdfs
    assert gdf_mcs.mcs_id.nunique() == ds.dims["mcs_id"], "same number of MCS"
    assert (
        mcs_mask_unique == np.r_[0, gdf_mcs_reid.mcs_id.unique() + 1]
    ).all(), "mask only contains re-IDed MCSs"
    assert (gdf_mcs.mcs_id.unique() == ds.mcs_id_orig - 1).all(), "MCS ID orig correct in ds"
    assert (gdf_mcs_reid.mcs_id.unique() == ds.mcs_id - 1).all(), "MCS ID correct in ds"

    # Save ds
    # <last_name>_WY<YYYY>_<DATA>_SAAG-MCS-mask-file.nc
    # DATA can either be OBS or WRF
    encoding: dict[Hashable, dict[str, Any]] = {"mcs_mask": {"zlib": True, "complevel": 5}}
    ds.to_netcdf(OUT_DIR / out_name_mosa, encoding=encoding)  # type: ignore[arg-type]


if __name__ == "__main__":
    if do_bench:
        from time import perf_counter_ns

        tic = perf_counter_ns()
        run_gdf(IN_DIR / "bench.parquet", bench=True)
        print(f"took {(perf_counter_ns() - tic) / 1e9:.1f} sec")

        raise SystemExit()

    import joblib

    joblib.Parallel(n_jobs=-2, verbose=1)(joblib.delayed(run_gdf)(fp) for fp in gdf_fps)
