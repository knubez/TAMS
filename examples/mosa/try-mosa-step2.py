"""
Try the MOSA runner for pre-processed data,
using a subset of the pre-processed files.

(Steps 2 _and_ 3.)
"""
from __future__ import annotations

import operator
from functools import reduce
from pathlib import Path
from typing import Any, Hashable

import numpy as np
import xarray as xr

from lib import _classify_cols, gdf_to_df, gdf_to_ds, re_id, run_wrf_preproced

base = Path("~/OneDrive/w/ERT-ARL/mosa").expanduser()
files = sorted((base / "mosa-pre-sample").glob("tb_rainrate*.parquet"))


# DataFrame output
gdf = run_wrf_preproced(files, id_="WRF")
df = gdf_to_df(gdf)

# Dataset output
ds_grid = (
    xr.open_dataset(base / "mosa-pre-sample/tb_rainrate_2010-06-01_01%3A00.nc")
    .rename_dims({"rlat": "y", "rlon": "x"})
    .squeeze()
)
ds_grid.lon.attrs.update(units="degree_east")
ds = gdf_to_ds(gdf, grid=ds_grid)

# Check `is_mcs` and crit consistency
crit_cols = [vn for vn in _classify_cols if vn != "is_mcs"]
assert (
    reduce(operator.add, [ds.isel(mcs_id=ds.is_mcs)[vn].astype(int) for vn in crit_cols])
    == len(crit_cols)
).all()
ds_non_mcs = ds.isel(mcs_id=~ds.is_mcs)
assert (
    reduce(operator.add, [ds_non_mcs[vn].astype(int) for vn in crit_cols]) < len(crit_cols)
).all()

# Save file
encoding: dict[Hashable, dict[str, Any]] = {"mcs_mask": {"zlib": True, "complevel": 5}}
ds.to_netcdf(base / "tams_mcs-mask-sample_nocomp.nc")
ds.to_netcdf(base / "tams_mcs-mask-sample.nc", encoding=encoding)

# Drop non-MCS
ds2 = ds.isel(mcs_id=ds.is_mcs)
assert ds2[_classify_cols].all().all()
assert set(gdf[gdf.is_mcs].mcs_id.unique() + 1) == set(ds2.mcs_id.values)
ds2 = ds2.drop_vars(_classify_cols)
ds2.to_netcdf(base / "tams_mcs-mask-sample_reduced_bad.nc", encoding=encoding)

# non-MCS are still in the mask! :(
assert (np.unique(ds2.mcs_mask) == np.r_[0, gdf.mcs_id.unique() + 1]).all()

#
# Drop non-MCS in gdf first, then create ds
#

gdf_mcs = gdf[gdf.is_mcs]
assert gdf_mcs.mcs_id.nunique() < gdf.mcs_id.nunique()

gdf_mcs_reid = gdf_mcs.assign(mcs_id=re_id(gdf_mcs), mcs_id_orig=gdf_mcs.mcs_id)
assert gdf_mcs_reid.groupby("mcs_id").nunique()["mcs_id_orig"].eq(1).all()

ds3 = gdf_to_ds(gdf_mcs_reid, grid=ds_grid)
assert ds3[_classify_cols].all().all()
ds3 = ds3.drop_vars(_classify_cols)

assert gdf_mcs.mcs_id.nunique() == ds3.dims["mcs_id"]
assert (np.unique(ds3.mcs_mask) == np.r_[0, gdf_mcs_reid.mcs_id.unique() + 1]).all()
assert (gdf_mcs.mcs_id.unique() == ds3.mcs_id_orig - 1).all()

ds3.to_netcdf(base / "tams_mcs-mask-sample_reduced.nc", encoding=encoding)
