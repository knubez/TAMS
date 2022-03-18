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

# TODO: sort by centroid lon value like below (optional somewhere?)
# fmt: off
with warnings.catch_warnings():
    warnings.filterwarnings(
        "ignore",
        category=UserWarning,
        message="Geometry is in a geographic CRS. Results from 'centroid' are likely incorrect.",
    )
    cs0 = (
        cs0
        .assign(x=cs0.geometry.centroid.x)
        .sort_values("x", ascending=False)
        .drop(columns="x")
    )
# fmt: on

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
        warnings.filterwarnings(
            "ignore", category=RuntimeWarning, message="invalid value encountered in intersection"
        )
        inter = cs1.intersection(cs0_i_poly)  # .dropna()
    inter = inter[~inter.is_empty]
    ov = inter.to_crs("EPSG:32663").area / cs0_area.iloc[i]
    # print(ov)
    res[i] = ov.to_dict()

pp(res)

fig, ax = plt.subplots(figsize=(16, 5))

cs0.plot(ax=ax, fc="blue", alpha=0.5)
cs1.plot(ax=ax, ec="green", fc="none", lw=2.5)

# UserWarning: Geometry is in a geographic CRS. Results from 'centroid' are likely incorrect.
# Use 'GeoSeries.to_crs()' to re-project geometries to a projected CRS before this operation.

with warnings.catch_warnings():
    warnings.filterwarnings(
        "ignore",
        category=UserWarning,
        message="Geometry is in a geographic CRS. Results from 'centroid' are likely incorrect.",
    )

    for i, (x, y) in enumerate(zip(cs0.centroid.x, cs0.centroid.y)):
        ax.text(x, y, i, c="blue", fontsize=14)

    for i, (x, y) in enumerate(zip(cs1.centroid.x, cs1.centroid.y)):
        ax.text(x, y, i, c="green", fontsize=14)

fig.tight_layout()

plt.show()
