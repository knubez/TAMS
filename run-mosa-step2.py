"""
Run TAMS on MOSA pre-processed data and produce output files (step 2)
- track
- classify CEs
- save df

Based on nb MOSA-2.ipynb
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from tams.mosa import run_wrf_preproced

IN_DIR = Path("/glade/scratch/zmoon/mosa-pre")
OUT_DIR = Path("/glade/scratch/zmoon/mosa2")  # TODO: just 'mosa' ?


#
# Get files
#


files = sorted(IN_DIR.glob("tb_rainrate*.parquet"))

print(f"{len(files)} total files")

print(files[0])
print("...")
print(files[-1])


# Split files into WYs
extra = []
wy_files = defaultdict(list)
for p in files:
    pre = p.name[:16]
    if pre in {"tb_rainrate_2010", "tb_rainrate_2011"}:
        wy_files[2011].append(p)
    elif pre in {"tb_rainrate_2015", "tb_rainrate_2016"}:
        wy_files[2016].append(p)
    elif pre in {"tb_rainrate_2018", "tb_rainrate_2019"}:
        wy_files[2019].append(p)
    else:
        extra.append(p)

assert not extra


#
# Run WYs
#


def run_wy(wy: int, files: list[Path], rt="df"):
    # Grid?
    if rt == "ds":
        import xarray as xr

        p_grid = (
            f"/glade/campaign/mmm/c3we/prein/SouthAmerica/MCS-Tracking/WY{wy}/WRF/"
            f"tb_rainrate_{wy - 1}-06-01_00:00.nc"
        )
        grid = xr.open_dataset(p_grid).rename_dims({"rlat": "y", "rlon": "x"}).squeeze()
    else:
        grid = None

    # Run
    ret = run_wrf_preproced(files, id_=f"WY{wy}", rt=rt, grid=grid)

    if rt == "df":
        # Only CEs associated to MCS (as defined by MOSA)
        ce = ret
        mcs = ce[ce.is_mcs].reset_index(drop=True).drop(columns=["is_mcs", "not_is_mcs_reason"])
        mcs

        # Save df
        mcs.to_csv(OUT_DIR / f"wrf_wy{wy}.csv.gz", index=False)

    elif rt == "ds":
        ds = ret

        # Save ds
        # <last_name>_WY<YYYY>_<DATA>_SAAG-MCS-mask-file.nc
        # DATA can either be OBS or WRF
        encoding = {"mcs_mask": {"zlib": True, "complevel": 5}}
        ds.to_netcdf(OUT_DIR / f"TAMS_WY{wy}_WRF_SAAG-MCS-mask-file_all.nc", encoding=encoding)  # type: ignore[arg-type]

        # Drop those not identified as MCSs using the MOSA criteria
        is_mcs = ds.is_mcs.to_series().groupby("mcs_id").agg(lambda x: x[~x.isnull()].unique())
        assert is_mcs.apply(len).eq(1).all()
        is_mcs = is_mcs.explode()
        ids = is_mcs[is_mcs].index
        ds2 = ds.sel(mcs_id=ids)
        assert ds2.is_mcs.all()
        ds2 = ds2.drop_vars(["is_mcs", "not_is_mcs_reason"])
        ds2.to_netcdf(OUT_DIR / f"TAMS_WY{wy}_WRF_SAAG-MCS-mask-file.nc", encoding=encoding)  # type: ignore[arg-type]

    else:
        raise ValueError(f"invalid `rt` {rt!r}")


if __name__ == "__main__":
    from functools import partial

    import joblib

    fn = partial(run_wy, rt="ds")

    joblib.Parallel(n_jobs=-2, verbose=1)(
        joblib.delayed(fn)(wy, files) for wy, files in wy_files.items()
    )
