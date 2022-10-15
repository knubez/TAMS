"""
Try the MOSA runner for preprocessed files
"""
from pathlib import Path

import xarray as xr

from tams.mosa import run_wrf_preproced

files = sorted(Path("~/Downloads/mosa-pre-sample").expanduser().glob("tb_rainrate*.parquet"))

# DataFrame output
df = run_wrf_preproced(files, id_="df")

# Dataset output
ds_grid = (
    xr.open_dataset("~/Downloads/mosa-pre-sample/tb_rainrate_2010-06-01_01%3A00.nc")
    .rename_dims({"rlat": "y", "rlon": "x"})
    .squeeze()
)
ds = run_wrf_preproced(files, rt="ds", grid=ds_grid, id_="ds")
