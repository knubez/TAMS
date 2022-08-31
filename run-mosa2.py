"""
Run on MOSA pre-processed data (step 2)
- track
- classify CEs
- save df

Based on nb MOSA-2.ipynb
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import joblib

from tams.mosa import run_wrf_preproced

IN_DIR = Path("/glade/scratch/zmoon/mosa-pre")
OUT_DIR = Path("/glade/scratch/zmoon/mosa2")


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

print(f"using {len(files)} files")


#
# Run WYs
#


def run_wy(wy: int, files: list[Path]):
    # Run
    ce = run_wrf_preproced(files)

    # Only CEs associated to MCS (as defined by MOSA)
    mcs = ce[ce.is_mcs].reset_index(drop=True).drop(columns=["is_mcs", "not_is_mcs_reason"])
    mcs

    # Save df
    mcs.to_csv(OUT_DIR / f"wrf_wy{wy}.csv.gz", index=False)


joblib.Parallel(n_jobs=-2, verbose=1)(
    joblib.delayed(run_wy)(wy, files) for wy, files in wy_files.items()
)
