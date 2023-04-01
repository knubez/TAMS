"""
Try the MOSA runner for pre-processed data,
using a subset of the pre-processed files.

(Steps 2 _and_ 3.)
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Hashable

import numpy as np
import xarray as xr

from lib import _classify_cols, gdf_to_df, gdf_to_ds, run_wrf_preproced


def re_id(ce):
    current_ids = sorted(ce.mcs_id.unique())
    to_new_id = {old_id: new_id for new_id, old_id in enumerate(current_ids)}
    return ce.mcs_id.map(to_new_id)


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
ds = gdf_to_ds(gdf, grid=ds_grid)

# Save file
encoding: dict[Hashable, dict[str, Any]] = {"mcs_mask": {"zlib": True, "complevel": 5}}
ds.to_netcdf(base / "tams_mcs-mask-sample_nocomp.nc")
ds.to_netcdf(base / "tams_mcs-mask-sample.nc", encoding=encoding)

# Drop non-MCS
is_mcs = ds.is_mcs.to_series().groupby("mcs_id").agg(lambda x: x[~x.isnull()].unique())
assert is_mcs.apply(len).eq(1).all()
is_mcs = is_mcs.explode()
ids = is_mcs[is_mcs].index
ds2 = ds.sel(mcs_id=ids)
assert ds2.is_mcs.all()
ds2 = ds2.drop_vars(_classify_cols)
ds2.to_netcdf(base / "tams_mcs-mask-sample_reduced.nc", encoding=encoding)

# non-MCS are still in the mask! :(
assert (np.unique(ds2.mcs_mask) == np.r_[0, gdf.mcs_id.unique() + 1]).all()

#
# Drop non-MCS in gdf first, then create ds
#

gdf_mcs = gdf[gdf.is_mcs]
assert gdf_mcs.mcs_id.nunique() < gdf.mcs_id.nunique()

gdf_mcs_reid = gdf_mcs.assign(mcs_id=re_id(gdf_mcs))
ds3 = gdf_to_ds(gdf_mcs_reid, grid=ds_grid)

assert gdf_mcs.mcs_id.nunique() == ds3.dims["mcs_id"]
assert (np.unique(ds3.mcs_mask) == np.r_[0, gdf_mcs_reid.mcs_id.unique() + 1]).all()
