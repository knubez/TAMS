import pandas as pd

from lib import load_preproc_zip

times, dfs = load_preproc_zip(
    "C:/Users/zmoon/OneDrive/w/ERT-ARL/mosa/mosa-pre_2010-08_wrf_v0.zip",
    kind="wrf",
)

assert times[0] == dfs[0].attrs["time"] == pd.Timestamp("2010-08-01 00:00:00")
print(dfs[0])

dt = pd.to_datetime(times).to_series().diff()
assert (dt[1:] == pd.Timedelta("1H")).all()
