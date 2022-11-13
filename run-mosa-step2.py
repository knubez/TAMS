"""
Run TAMS on MOSA pre-processed data and produce output files (step 2)
- track
- classify CEs
- save gdf

Based on nb MOSA-2.ipynb
"""
from __future__ import annotations

import warnings
from collections import defaultdict
from pathlib import Path

from tams.mosa import run_wrf_preproced

IN_DIR = Path("/glade/scratch/zmoon/mosa-pre")
OUT_DIR = Path("/glade/scratch/zmoon/mosa2")  # TODO: just 'mosa' ?


#
# Get files
#

# WRF
files = sorted(IN_DIR.glob("tb_rainrate_*.parquet"))

print(f"{len(files)} total files (WRF)")

print(files[0])
print("...")
print(files[-1])

# Split files into WYs
extra = []
wy_files = defaultdict(list)
for p in files:
    pre = p.name[: len("tb_rainrate_2010")]
    if pre in {"tb_rainrate_2010", "tb_rainrate_2011"}:
        wy_files[2011].append(p)
    elif pre in {"tb_rainrate_2015", "tb_rainrate_2016"}:
        wy_files[2016].append(p)
    elif pre in {"tb_rainrate_2018", "tb_rainrate_2019"}:
        wy_files[2019].append(p)
    else:
        extra.append(p)

assert not extra
wy_files_wrf = wy_files

# GPM
files = sorted(IN_DIR.glob("merg_*.parquet"))

print(f"{len(files)} total files (GPM)")

print(files[0])
print("...")
print(files[-1])

# Split files into WYs
extra = []
wy_files = defaultdict(list)
for p in files:
    pre = p.name[: len("merg_2010")]
    if pre in {"merg_2010", "merg_2011"}:
        wy_files[2011].append(p)
    elif pre in {"merg_2015", "merg_2016"}:
        wy_files[2016].append(p)
    elif pre in {"merg_2018", "merg_2019"}:
        wy_files[2019].append(p)
    else:
        extra.append(p)

assert not extra
wy_files_gpm = wy_files


#
# Run WYs
#


def run_wy(wy: int, files: list[Path]):
    # Run
    gdf = run_wrf_preproced(files, id_=f"WY{wy}")

    # Save
    f0n = files[0].name
    if f0n.startswith("tb_rainrate_"):
        which = "wrf"
    elif f0n.startswith("merg_"):
        which = "gpm"
    else:
        raise ValueError("Unexpected file name {f0n!r}, unable to determine WRF or GPM.")
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*initial implementation of Parquet.*")
        gdf.to_parquet(OUT_DIR / f"{which}_wy{wy}.parquet")


if __name__ == "__main__":
    import itertools
    from functools import partial

    import joblib

    fn = partial(run_wy, rt="ds")

    joblib.Parallel(n_jobs=-2, verbose=1)(
        joblib.delayed(fn)(wy, files)
        for wy, files in itertools.chain(wy_files_wrf.items(), wy_files_gpm.items())
    )
