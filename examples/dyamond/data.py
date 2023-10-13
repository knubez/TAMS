import datetime
import re
from pathlib import Path

import numpy as np
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

    start = datetime.datetime(2016, 8, 1) if season == "Summer" else datetime.datetime(2020, 1, 20)
    time_should_be = [start]
    for _ in range(40 * 24 - 1):
        time_should_be.append(time_should_be[-1] + datetime.timedelta(hours=1))
    assert time_should_be[-1] - time_should_be[0] == datetime.timedelta(days=39, hours=23)

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

        ymdh = re.search(r"[0-9]{10}", fp.stem).group()
        t_file = datetime.datetime.strptime(ymdh, r"%Y%m%d%H")
        print(pad, "t path:", t_file)

        assert ds.dims["lon"] == 3600, "0.1 deg"
        lona, lonb = -179.95, 179.95
        if model == "OBS" and season.lower() == "summer":
            lata, latb = -89.95, 89.95
            nlat = 1800
        else:
            lata, latb = -59.95, 59.95
            nlat = 1200
        assert ds.dims["lat"] == nlat
        lon, lat = ds["lon"], ds["lat"]
        assert lat.dims == ("lat",)
        try:
            assert lat.values[0] == lata and lat.values[-1] == latb
        except AssertionError:
            print(pad, "unexpected lat range:", lat.values[0], "...", lat.values[-1])
        assert lon.dims == ("lon",)
        assert lon.values[0] == lona and np.isclose(lon.values[-1], lonb)

        dvs = sorted(ds.data_vars)
        print(pad, "dvs:", dvs)

        rn = vn_map[season.lower()][model]
        assert rn.keys() <= set(dvs), "remap"

        # Check which times we have
        t_files = []
        for fp in files:
            ymdh = re.search(r"[0-9]{10}", fp.stem).group()
            t_file = datetime.datetime.strptime(ymdh, r"%Y%m%d%H")
            t_files.append(t_file)
        t_miss = [t for t in time_should_be if t not in t_files]
        if t_miss:
            print(pad, f"missing files for these {len(t_miss)} times:")
            for t in t_miss:
                print(pad, "-", t.strftime(r"%Y-%m-%d %H"))

# TODO: idealized cases data ('idealized_cases')
