"""
Try the MOSA runner for pre-processed data,
using a subset of the pre-processed files.
"""
from pathlib import Path

import xarray as xr

from tams.mosa import run_wrf_preproced

base = Path("~/OneDrive/w/ERT-ARL/mosa").expanduser()
files = sorted((base / "mosa-pre-sample").glob("tb_rainrate*.parquet"))

# DataFrame output
df = run_wrf_preproced(files, id_="df")

# Dataset output
ds_grid = (
    xr.open_dataset(base / "mosa-pre-sample/tb_rainrate_2010-06-01_01%3A00.nc")
    .rename_dims({"rlat": "y", "rlon": "x"})
    .squeeze()
)
ds = run_wrf_preproced(files, rt="ds", grid=ds_grid, id_="ds")
