import warnings
from pprint import pp

import matplotlib.pyplot as plt

import tams

plt.close("all")


r = tams.load_example_ir()

tb = tams.tb_from_ir(r, ch=9)

tb0 = tb.isel(time=0)
tb1 = tb.isel(time=1)

cs0 = tams.identify(tb0)
cs1 = tams.identify(tb1)

# TODO: sort by centroid lon value

# For each in cs0, check overlap with all in cs1

cs0_area = cs0.to_crs("EPSG:32663").area
res = {}
for i in range(len(cs0)):
    cs0_i = cs0.iloc[i : i + 1]  # slicing preserves GeoDataFrame type
    cs0_i_poly = cs0_i.values[0][0]
    with warnings.catch_warnings():
        # We get
        # pygeos\set_operations.py:129: RuntimeWarning: invalid value encountered in intersection
        # when an empty intersection is found
        inter = cs1.intersection(cs0_i_poly)  # .dropna()
    inter = inter[~inter.is_empty]
    ov = inter.to_crs("EPSG:32663").area / cs0_area.iloc[i]
    # print(ov)
    res[i] = ov.to_dict()

pp(res)

fig, ax = plt.subplots(figsize=(16, 5))

cs0.plot(ax=ax, fc="blue", alpha=0.5)
cs1.plot(ax=ax, ec="green", fc="none", lw=2.5)

for i, (x, y) in enumerate(zip(cs0.centroid.x, cs0.centroid.y)):
    ax.text(x, y, i, c="blue", fontsize=14)

for i, (x, y) in enumerate(zip(cs1.centroid.x, cs1.centroid.y)):
    ax.text(x, y, i, c="green", fontsize=14)

fig.tight_layout()

plt.show()
