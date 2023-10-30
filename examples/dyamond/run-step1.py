"""
Pre-process DYAMOND files and save Parquet files to TAR archive.

NOTE: requires `pyarrow` to save the GeoDataFrames
"""

from lib import BASE_DIR_OUT_PRE, iter_input_paths, preproc_file

BASE_DIR_OUT_PRE.mkdir(exist_ok=True)

parallel = True
fn = preproc_file

files = list(iter_input_paths())

print(f"{len(files)} files")
print(files[0])
print("...")
print(files[-1])

if parallel:
    import joblib

    joblib.Parallel(n_jobs=-2, verbose=10)(joblib.delayed(fn)(fp) for fp in files)
else:
    for fp in files:
        fn(fp)
