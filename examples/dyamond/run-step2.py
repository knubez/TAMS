"""
Run TAMS on MOSA pre-processed data and produce output files (step 2)
- track
- classify CEs
- save gdf
"""
from lib import BASE_DIR_OUT, get_preproced_paths, run

parallel = True


def func(key, files):
    season, model = key
    id_ = f"{season}__{model}"
    ce = run(files, id_=id_)
    ce.to_parquet(BASE_DIR_OUT / f"{id_}.parquet", schema_version="0.4.0")


cases = get_preproced_paths()

print(f"{len(cases)} cases")
print("\n".join(sorted(cases)))

if parallel:
    import joblib

    joblib.Parallel(n_jobs=-2, verbose=10)(
        joblib.delayed(func)(key, files) for key, files in cases.items()
    )
else:
    for key, files in cases.items():
        func(key, files)
