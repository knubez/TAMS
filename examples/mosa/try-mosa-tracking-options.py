"""
Try the MOSA runner for pre-processed data,
using a subset of the pre-processed files,
Comparing the different tracking options.
"""
from __future__ import annotations

from pathlib import Path
from time import perf_counter_ns

import pandas as pd

from lib import run_wrf_preproced

base = Path("~/OneDrive/w/ERT-ARL/mosa").expanduser()
files = sorted((base / "mosa-pre-sample").glob("tb_rainrate*.parquet"))

# Tracking options
opts = {
    "b": dict(largest=False, look="b"),
    "bl": dict(largest=True, look="b"),
    "f": dict(largest=False, look="f"),
    "fl": dict(largest=True, look="f"),
    "flm": dict(largest=True, look="f", overlap_norm="min"),
}
gdfs = {}
for id_, kws in opts.items():
    tic = perf_counter_ns()
    gdf = run_wrf_preproced(files, id_=f"WRF {id_}", track_kws=kws).assign(run_id=id_)
    toc = perf_counter_ns()
    gdfs[id_] = gdf.assign(runtime=toc - tic)


# Stats
def pct(x):
    "Percent True"
    return x.sum() / len(x)


gdf_c = pd.concat(gdfs.values())
stats = pd.concat(
    [
        # gdf_c[["area_km2", "run_id"]].groupby("run_id").agg(["min", "mean", "max"]),
        gdf_c[["is_mcs", "run_id"]].groupby("run_id").agg(["sum", pct]),
        gdf_c[["runtime", "run_id"]]
        .groupby("run_id")
        .agg("first")
        .rename(columns={"runtime": ("runtime", "_")}),
        gdf_c[["mcs_id", "run_id"]]
        .groupby("run_id")
        .nunique()
        .rename(columns={"mcs_id": ("mcs_id", "nunique")}),
    ],
    axis="columns",
)
