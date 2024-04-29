"""
Convert from CE gdf output to mask file
"""
from lib import BASE_DIR_OUT, post

parallel = True
new_data = True

func = post

gdf_fps = sorted(BASE_DIR_OUT.glob("*.parquet"))

if new_data:
    gdf_fps = [
        fp for fp in gdf_fps if fp.stem.split("__")[1] in {"OBSv7".lower(), "SCREAMv1".lower()}
    ]

print(f"{len(gdf_fps)} files:")
print("\n".join(f"- {fp.as_posix()}" for fp in gdf_fps))

if parallel:
    import joblib

    joblib.Parallel(n_jobs=-2, verbose=10)(joblib.delayed(func)(fp) for fp in gdf_fps)
else:
    for fp in gdf_fps:
        func(fp)
