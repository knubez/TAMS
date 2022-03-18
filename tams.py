"""
TAMS
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import xarray as xr

if TYPE_CHECKING:
    import geopandas as gpd


_tb_from_ir_coeffs: dict[int, tuple[float, float, float]] = {
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
    in channel `ch` of the EUMETSAT MSG SEVIRI instrument.

    Reference: http://www.eumetrain.org/data/2/204/204.pdf page 13

    https://www.eumetsat.int/seviri

    Parameters
    ----------
    r : array-like
        Radiance. Units: m2 m-2 sr-1 (cm-1)-1
    ch
        Channel number, in 4--11.

    Returns
    -------
    tb
        Brightness temperature (same type as `r`)
    """
    if ch not in range(4, 12):
        raise ValueError("channel must be in 4--11")

    c1 = 1.19104e-5
    c2 = 1.43877

    vc, a, b = _tb_from_ir_coeffs[ch]

    tb = (c2 * vc / np.log((c1 * vc**3) / r + 1) - b) / a

    if isinstance(r, xr.DataArray):
        tb.attrs.update(units="K", long_name="Brightness temperature")

    return tb


def contours(x: xr.DataArray, value: float) -> list[np.ndarray]:
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
    # TODO: have this return GDF instead
    # TODO: allowing specifying `crs`, `method`, shapely options (buffer, convex-hull), ...
    import matplotlib.pyplot as plt

    assert x.ndim == 2, "this is for a single image"
    with plt.ioff():  # requires mpl 3.4
        fig = plt.figure()
        cs = x.plot.contour(x="lon", y="lat", levels=[value])

    plt.close(fig)
    assert len(cs.allsegs) == 1, "only one level"

    return cs.allsegs[0]


def _contours_to_gdf(cs: list[np.ndarray]) -> gpd.GeoDataFrame:
    from geopandas import GeoDataFrame
    from shapely.geometry.polygon import LinearRing, orient

    polys = []
    for c in cs:
        x, y = c.T
        r = LinearRing(zip(x, y))
        p0 = r.convex_hull  # TODO: optional, also add buffer option
        p = orient(p0)  # -> counter-clockwise
        polys.append(p)

    return GeoDataFrame(geometry=polys, crs="EPSG:4326")
    # ^ This crs indicates input in degrees


def _data_in_contours_sjoin(
    data: xr.DataArray | xr.Dataset,
    contours: gpd.GeoDataFrame,
    *,
    agg=("mean", "std", "count"),
) -> gpd.GeoDataFrame:
    """Compute stats on `data` within `contours` using :func:`~geopandas.tools.sjoin`.

    `data` must have ``'lat'`` and ``'lon'`` variables.
    """
    import geopandas as gpd

    # Detect variables to run the agg on
    if isinstance(data, xr.DataArray):
        varnames = [data.name]
    elif isinstance(data, xr.Dataset):
        # varnames = [vn for vn in field.variables if vn not in {"lat", "lon"}]
        raise NotImplementedError
    else:
        raise TypeError

    # Convert possibly-2-D data to GeoDataFrame of points
    data_df = data.to_dataframe().reset_index(drop=True)
    lat = data_df["lat"].values
    lon = data_df["lon"].values
    geom = gpd.points_from_xy(lon, lat, crs="EPSG:4326")  # can be slow with many points
    points = gpd.GeoDataFrame(data_df, geometry=geom)

    # Determine which contour (if any) each point is inside
    points = points.sjoin(contours, predicate="within", how="left", rsuffix="contour")
    points = points.dropna().convert_dtypes()
    points["lat"] = points.geometry.y
    points["lon"] = points.geometry.x

    # Aggregate points inside contour
    # TODO: a way to do this without groupby loop?
    new_data_ = {}
    for i, g in points.groupby("index_contour"):
        r = g[varnames].agg(agg).T  # columns: aggs; rows: variables
        new_data_[i] = r
    new_data = pd.concat(new_data_).convert_dtypes()

    # Convert to standard (non-multi) index and str columns
    new_data = new_data.unstack()  # multi index -> (variable, agg) columns
    new_data.columns = ["_".join(s for s in tup) for tup in new_data.columns]

    # Merge with contours gdf
    contours = contours.merge(new_data, left_index=True, right_index=True, how="left")

    return contours


def _data_in_contours_regionmask(
    data: xr.DataArray | xr.Dataset,
    contours: gpd.GeoDataFrame,
    *,
    agg=("mean", "std", "count"),
) -> gpd.GeoDataFrame:
    import regionmask

    # TODO: DRY? (much of this fn is same as other one; create user-facing fn to use either)
    if isinstance(data, xr.DataArray):
        varnames = [data.name]
    elif isinstance(data, xr.Dataset):
        raise NotImplementedError
    else:
        raise TypeError

    # Form regionmask(s)
    shapes = contours[["geometry"]]
    regions = regionmask.from_geopandas(shapes)
    mask = regions.mask(data)  # works but takes long (though shorter with pygeos)!

    # Aggregate points inside contour
    new_data_ = {
        i: data.where(mask == i).to_dataframe()[varnames].dropna().agg(agg).T
        for i in regions.numbers
    }
    new_data = pd.concat(new_data_).convert_dtypes()
    # TODO: also try with xarray methods instead of going through pandas
    # TODO: try with xarray groupby

    # Convert to standard (non-multi) index and str columns
    new_data = new_data.unstack()  # multi index -> (variable, agg) columns
    new_data.columns = ["_".join(s for s in tup) for tup in new_data.columns]

    # Merge with contours gdf
    contours = contours.merge(new_data, left_index=True, right_index=True, how="left")

    return contours


def _size_filter_contours(
    cs235: gpd.GeoDataFrame,
    cs219: gpd.GeoDataFrame,
    *,
    debug=True,
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """Compute areas and use to filter both sets of contours."""

    # Drop small 235s
    cs235["area_km2"] = cs235.to_crs("EPSG:32663").area / 10**6
    # ^ This crs is equidistant cylindrical
    big_enough = cs235.area_km2 >= 4000
    if debug:
        print(
            f"{big_enough.value_counts()[True] / big_enough.size * 100:.1f}% "
            " of 235s are big enough"
        )
    cs235 = cs235[big_enough].reset_index(drop=True)

    # Identify indices of 219s inside 235s
    # Note that some 235s might not have any 219s inside
    a = cs235.sjoin(cs219, predicate="contains", how="left").reset_index()
    # ^ gives an Int64 index with duplicated values, for each 219 inside a certain 235
    i219s = {  # convert to list
        i235: g.index_right.astype(int).to_list()
        for i235, g in a.groupby("index")
        if not g.index_right.isna().all()
    }

    # Check 219 area sum inside the 235
    sum219s = {
        i235: cs219.iloc[i219s.get(i235, [])].to_crs("EPSG:32663").area.sum() / 10**6
        for i235 in cs235.index
    }
    cs235["area219_km2"] = pd.Series(sum219s)
    big_enough = cs235.area219_km2 >= 4000
    if debug:
        print(
            f"{big_enough.value_counts()[True] / big_enough.size * 100:.1f}% "
            "of big-enough 235s have enough 219 area"
        )
    cs235 = cs235[big_enough].reset_index(drop=True)

    # TODO: store 219 inds in the 235 df? optional output?

    return cs235, cs219


def identify(x, based_on="ctt") -> gpd.GeoDataFrame:
    """Identify clouds."""
    if based_on != "ctt":
        raise NotImplementedError

    cs235 = _contours_to_gdf(contours(x, 235))
    cs219 = _contours_to_gdf(contours(x, 219))

    cs235, _ = _size_filter_contours(cs235, cs219)

    return cs235


def load_example_ir() -> xr.DataArray:
    """Load the example IR radiance data (ch9) as a DataArray."""

    ds = xr.open_dataset("Satellite_data.nc").rename_dims(
        {"num_rows_vis_ir": "y", "num_columns_vis_ir": "x"}
    )

    ds.lon.attrs.update(long_name="Longitude")
    ds.lat.attrs.update(long_name="Latitude")

    return ds.ch9


def load_example_tb() -> xr.DataArray:
    """Load the example derived brightness temperature data as a DataArray,
    by first invoking :func:`load_example_ir` and then applying :func:`tb_from_ir`.
    """

    r = load_example_ir()

    return tb_from_ir(r, ch=9)


if __name__ == "__main__":
    import cartopy.crs as ccrs
    import matplotlib.pyplot as plt
    import regionmask

    r = load_example_ir().isel(time=0)

    tb = tb_from_ir(r, ch=9)

    tran = ccrs.PlateCarree()
    proj = ccrs.Mercator()
    fig, ax = plt.subplots(subplot_kw=dict(projection=proj))

    # tb.plot(x="lon", y="lat", cmap="gray_r", ax=ax)
    cs = contours(tb, 235)
    cs = sorted(cs, key=len, reverse=True)  # [:30]
    for c in cs:
        ax.plot(c[:, 0], c[:, 1], "g", transform=tran)

    cs235 = _contours_to_gdf(cs)
    cs219 = _contours_to_gdf(contours(tb, 219))

    cs235, cs219 = _size_filter_contours(cs235, cs219)

    # Trying regionmask
    shapes = cs235[["geometry"]]
    regions = regionmask.from_geopandas(shapes)
    mask = regions.mask(tb)  # works but takes long (though shorter with pygeos)!

    regions.plot(ax=ax)

    # tb.where(mask >= 0).plot.pcolormesh(ax=ax, transform=tran)  # takes long
    tb.where(mask >= 0).plot.pcolormesh(size=4, aspect=2)

    plt.show()
