"""
TAMS
"""
import numpy as np

_tb_from_ir_coeffs = {
    4: (2569.094, 0.9959, 3.471),
    5: (1598.566, 0.9963, 2.219),
    6: (1362.142, 0.9991, 0.485),
    7: (1149.083, 0.9996, 0.181),
    8: (1034.345, 0.9999, 0.060),
    9: (930.659, 0.9983, 0.627),
    10: (839.661, 0.9988, 0.397),
    11: (752.381, 0.9981, 0.576),
}


def tb_from_ir(r, ch: int):
    """Compute brightness temperature from IR satellite radiances (`r`)
    in channel `ch`.

    Reference: http://www.eumetrain.org/data/2/204/204.pdf page 13

    Parameters
    ----------
    r : array-like
        Radiance. Units: m2 m-2 sr-1 (cm-1)-1
    ch
        Channel number, in 4--11.
    """
    if ch not in range(4, 12):
        raise ValueError("channel must be in 4--11")

    c1 = 1.19104e-5
    c2 = 1.43877

    vc, a, b = _tb_from_ir_coeffs[ch]

    tb = (c2 * vc / np.log((c1 * vc ** 3) / r + 1) - b) / a

    tb.attrs.update(units="K", long_name="Brightness temperature")

    return tb


def contours(x, value: float):
    """Find contour definitions for 2-D data `x` at value `value`.

    Parameters
    ----------
    x : xarray.DataArray
        Data to be contoured.
        Currently needs to have 'lat' and 'lon' coordinates.

    Returns
    -------
    list of numpy.ndarray
        List of 2-D arrays describing contours.
        The arrays are shape (n, 2); each row is a coordinate pair.
    """
    import matplotlib.pyplot as plt

    assert x.ndim == 2, "this is for a single image"
    with plt.ioff():  # requires mpl 3.4
        fig = plt.figure()
        cs = x.plot.contour(x="lon", y="lat", levels=[value])

    plt.close(fig)
    assert len(cs.allsegs) == 1, "only one level"

    return cs.allsegs[0]


def _contours_to_gdf(cs):
    from geopandas import GeoDataFrame
    from shapely.geometry.polygon import LinearRing, orient

    polys = []
    for c in cs:
        x, y = c.T
        r = LinearRing(zip(x, y))
        p0 = r.convex_hull
        p = orient(p0)  # -> counter-clockwise
        polys.append(p)

    return GeoDataFrame(geometry=polys)  # TODO: crs


def load_example_ir():
    """Load the example radiance data (ch9) as a DataArray."""
    import xarray as xr

    ds = xr.open_dataset("Satellite_data.nc").rename_dims(
        {"num_rows_vis_ir": "y", "num_columns_vis_ir": "x"}
    )

    ds.lon.attrs.update(long_name="Longitude")
    ds.lat.attrs.update(long_name="Latitude")

    return ds.ch9


if __name__ == "__main__":
    import cartopy.crs as ccrs
    import matplotlib.pyplot as plt
    import regionmask

    # import geopandas as gpd
    # from shapely.geometry import Point

    r = load_example_ir().isel(time=0)

    tb = tb_from_ir(r, ch=9)

    tran = ccrs.PlateCarree()
    proj = ccrs.Mercator()
    fig, ax = plt.subplots(subplot_kw=dict(projection=proj))

    # tb.plot(x="lon", y="lat", cmap="gray_r", ax=ax)
    cs = contours(tb, 235)
    cs = sorted(cs, key=len, reverse=True)[:30]
    for c in cs:
        ax.plot(c[:, 0], c[:, 1], "g", transform=tran)

    shapes = _contours_to_gdf(cs)
    regions = regionmask.from_geopandas(shapes)
    mask = regions.mask(tb)  # works but takes long (though shorter with pygeos)!

    regions.plot(ax=ax)

    # tb.where(mask >= 0).plot.pcolormesh(ax=ax, transform=tran)  # takes long
    tb.where(mask >= 0).plot.pcolormesh(size=4, aspect=2)

    # # Spatial join from GeoPandas
    # points = tb.to_dataframe().drop("ch9", axis="columns").reset_index(drop=True)
    # points["coords"] = list(zip(points.lon, points.lat))
    # points["coords"] = points.coords.apply(Point)
    # points = gpd.GeoDataFrame(points, geometry="coords")
    # # or `gpd.points_from_xy(df.lon, df.lat)` but was slower?
    # in_polys = gpd.tools.sjoin(points, shapes, predicate="within", how="left")

    plt.show()
