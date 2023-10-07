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


def _make_vn_map():
    def _to_map(s):
        lines = s.splitlines()
        n = len(lines)
        assert n % 3 == 0
        m = {}
        for i in range(3, n, 3):
            model, pr, olr = lines[i : i + 3]
            if model == "ARPEGE-NH":
                model = "ARPEGE"
            m[model] = {pr: "pr", olr: "olr"}
        return m

    d = {}

    # Summer
    s = """\
Model
Precipitation
OLR
ARPEGE-NH
param8.1.0
ttr
FV3
pr
flut
IFS
tp
ttr
MPAS
pr
olrtoa
NICAM
sa_tppn
sa_lwu_toa
SAM
Precac
LWNTA
UM
precipitation_flux
toa_outgoing_longwave_flux"""
    d["summer"] = _to_map(s)

    # Winter
    s = """\
Model
Precipitation
OLR
ARPEGE-NH
pr
rlt
GEOS
pr
rlut
GRIST
pr
rlt
ICON
pr
rlut
IFS
pracc
rltacc
MPAS
pr
rltacc
NICAM
sa_tppn
sa_lwu_toa
SAM
pracc
rltacc
SCREAM
pr
rlt
UM
pr
rlut
XSHiELD
pr
rlut"""
    d["winter"] = _to_map(s)

    for season in d:
        d[season]["OBS"] = {"precipitationCal": "pr", "Tb": "tb"}

    return d


vn_map = _make_vn_map()

print(vn_map)

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

        fp = files[0]
        fn = fp.name
        print(pad, "first:", fn)
        ds = xr.open_dataset(fp)
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

        dvs = sorted(ds.data_vars)
        print(pad, dvs)

        rn = vn_map[season.lower()][model]
        assert rn.keys() <= set(dvs), "remap"

# TODO: idealized cases data ('idealized_cases')
