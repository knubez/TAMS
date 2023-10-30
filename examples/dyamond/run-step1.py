"""
Pre-process DYAMOND files and save Parquet files to TAR archive.
"""
from functools import partial

from lib import BASE_DIR_OUT_PRE, iter_input_paths, preproc_file

BASE_DIR_OUT_PRE.mkdir(exist_ok=True)

parallel = True
func = partial(preproc_file, overwrite=False)

files = list(iter_input_paths())

print(f"{len(files)} files")
print(files[0])
print("...")
print(files[-1])

if parallel:
    import joblib

    joblib.Parallel(n_jobs=-2, verbose=10)(joblib.delayed(func)(fp) for fp in files)
else:
    for fp in files:
        func(fp)
