"""
Pre-process MOSA files and save

NOTE: requires `pyarrow` to save the GeoDataFrames

warning: no pr data in contours for tb_rainrate_2015-06-24_21:00.nc
warning: no pr data in contours for tb_rainrate_2015-07-08_19:00.nc

NOTE: took 8.7 min for one WY using 19 procs
"""
from functools import partial
from pathlib import Path

from tams.mosa import BASE_DIR, preproc_wrf_file

# Constants
out_dir = Path("/glade/scratch/zmoon/mosa-pre")
parallel = True
wys = [2011, 2016, 2019]

assert out_dir.is_dir()
preproc = partial(preproc_wrf_file, out_dir=out_dir)

if parallel:
    import joblib

for wy in wys:
    files = sorted((BASE_DIR / f"WY{wy}" / "WRF").glob("tb_rainrate_*.nc"))
    print(files[0])
    print("...")
    print(files[-1], f"({len(files)} total)")

    # Remember that the first file has no CTT!
    files = files[1:]

    if parallel:
        joblib.Parallel(n_jobs=-2, verbose=10)(joblib.delayed(preproc)(fp) for fp in files)
    else:
        for fp in files:
            preproc(fp)
