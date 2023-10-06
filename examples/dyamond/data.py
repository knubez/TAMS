import datetime
import re
from pathlib import Path

import xarray as xr

base = Path("/glade/campaign/mmm/c3we/prein/Papers/2023_Zhe-MCSMIP")
# Under here we have 'Summer' and 'Winter' dirs
# and then model dirs with 'olr_pcp_instantaneous' subdirs
# that have the nc files, one for each hour
# e.g.
# /glade/campaign/mmm/c3we/prein/Papers/2023_Zhe-MCSMIP/Winter/UM/olr_pcp_instantaneous/pr_rlut_um_winter_2020012007.nc

for season in ["Summer", "Winter"]:
    print(season)
    d = base / season
    assert d.is_dir()

    model_dirs = sorted(d.glob("*"))
    for model_dir in model_dirs:
        model = model_dir.name
        print(" ", model)
        (subdir,) = list(model_dir.glob("*"))  # just one
        assert subdir.name in {"olr_pcp", "olr_pcp_instantaneous"}

        files = sorted(subdir.glob("*"))
        # They don't all have the same number of files
        # but the most common number is 960 (40 days)
        pad = " " * 3
        print(pad, "n =", len(files))

        print(pad, "first:", files[0].name)
        ds = xr.open_dataset(files[0])
        if model == "MPAS":
            ds = ds.rename(xtime="time")
        if model == "OBS" and season == "Summer":
            assert ds.dims["time"] == 2
            ds = ds.isel(time=0).expand_dims("time", axis=0)
        assert {"time", "lon", "lat"} <= set(ds.dims)  # some also have 'bnds'
        assert ds.dims["time"] == 1
        print(pad, "t data:", ds.time.values[0])

        ymdh = re.search(r"[0-9]{10}", files[0].stem).group()
        t_file = datetime.datetime.strptime(ymdh, r"%Y%m%d%H")
        print(pad, "t file:", t_file)

        print(pad, sorted(ds.data_vars))

# TODO: idealized cases data ('idealized_cases')
