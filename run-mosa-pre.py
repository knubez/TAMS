"""
Pre-process MOSA files and save

NOTE: requires `pyarrow` to save the GeoDataFrames

warning: no pr data in contours for tb_rainrate_2015-06-24_21:00.nc
warning: no pr data in contours for tb_rainrate_2015-07-08_19:00.nc
"""
from functools import partial
from pathlib import Path

from tams.mosa import BASE_DIR, preproc_wrf_file

# files = sorted((BASE_DIR / "WY2011" / "WRF").glob("tb_rainrate_*.nc"))
# files = sorted((BASE_DIR / "WY2016" / "WRF").glob("tb_rainrate_*.nc"))
files = sorted((BASE_DIR / "WY2019" / "WRF").glob("tb_rainrate_*.nc"))
print(files[0])
print("...")
print(files[-1], f"({len(files)} total)")
# Remember first has no CTT!

# files = files[1:11]  # testing
files = files[1:]

out_dir = Path("/glade/scratch/zmoon/mosa-pre")
assert out_dir.is_dir()

parallel = True
preproc = partial(preproc_wrf_file, out_dir=out_dir)

if parallel:
    import joblib

    joblib.Parallel(n_jobs=-2, verbose=10)(joblib.delayed(preproc)(fp) for fp in files)
    # 8.7 min with 19 procs
else:
    for fp in files:
        preproc(fp)
