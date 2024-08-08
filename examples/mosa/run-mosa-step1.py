"""
Pre-process MOSA files and save

NOTE: requires `pyarrow` to save the GeoDataFrames

warning: no pr data in contours for tb_rainrate_2015-06-24_21:00.nc
warning: no pr data in contours for tb_rainrate_2015-07-08_19:00.nc

NOTE: took 8.7 min for one WY using 19 procs
"""

from functools import partial
from pathlib import Path
from time import perf_counter_ns

import pandas as pd

from lib import BASE_DIR, preproc_gpm_file, preproc_wrf_file

# Constants
out_dir = Path("/glade/derecho/scratch/zmoon/mosa/pre")
parallel = True
do_wrf = True
do_gpm = True
do_bench = True
wys = [2011, 2016, 2019]

assert out_dir.is_dir()

if parallel:
    import joblib


def proc(fn, files):
    if parallel:
        joblib.Parallel(n_jobs=-2, verbose=10)(joblib.delayed(fn)(fp) for fp in files)
    else:
        for fp in files:
            fn(fp)


if do_bench:
    # WRF 1--7 Nov 2018
    times = pd.date_range("2018-11-01", "2018-11-07 23:00", freq="h")
    files = [BASE_DIR / "WY2019" / "WRF" / f"tb_rainrate_{t:%Y-%m-%d_%H}:00.nc" for t in times]

    print(files[0])
    print("...")
    print(files[-1], f"({len(files)} total)")

    fn = partial(preproc_wrf_file, out_dir=out_dir)

    tic = perf_counter_ns()
    proc(fn, files)
    print(f"took {(perf_counter_ns() - tic) / 1e9:.1f} s")

    raise SystemExit()


# WRF
if do_wrf:
    fn = partial(preproc_wrf_file, out_dir=out_dir)
    for wy in wys:
        files = sorted((BASE_DIR / f"WY{wy}" / "WRF").glob("tb_rainrate_*.nc"))
        print(files[0])
        print("...")
        print(files[-1], f"({len(files)} total)")

        # Remember that the first file has no CTT!
        files = files[1:]

        proc(fn, files)


# GPM
if do_gpm:
    fn = partial(preproc_gpm_file, out_dir=out_dir)
    for wy in wys:
        # The WRF file sets start in June of previous year
        ts = pd.date_range(f"{wy - 1}/06/01", f"{wy}/06/01", freq="h")[:-1]
        rfns = ts.strftime(r"%Y") + "/merg_" + ts.strftime(r"%Y%m%d%H") + "_4km-pixel.nc"
        files = [BASE_DIR / "GPM" / rfn for rfn in rfns]
        print(files[0])
        print("...")
        print(files[-1], f"({len(files)} total)")

        # Don't need to drop first file

        proc(fn, files)
