"""
Core routines that make up the TAMS algorithm.
"""

from __future__ import annotations

import functools
import logging
import warnings
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import xarray as xr

from .util import _the_unique, sort_ew

if TYPE_CHECKING:
    import geopandas
    import matplotlib
    import numpy
    import pandas
    import shapely
    import xarray


logger = logging.getLogger(__name__)


def contours(
    x: xarray.DataArray,
    value: float,
    *,
    unstructured: bool | None = None,
    **kwargs,
) -> list[numpy.ndarray]:
    """Find contour definitions for data `x` at value `value`.

    Parameters
    ----------
    x
        Data to be contoured.
        Currently needs to have ``'lat'`` and ``'lon'`` coordinates.
        The array should be 2-D if structured, 1-D if unstructured.
    value
        Find contours where `x` has this value.
    unstructured
        Whether the grid of `x` is unstructured (e.g. MPAS native output).
        Default: assume unstructured if `x` is 1-D.

    Returns
    -------
    :
        List of 2-D arrays describing contours.
        The arrays are shape ``(n, 2)``; each row is a coordinate pair.
    """
    if x.isnull().all():
        raise ValueError("Input array `x` is all null (e.g. NaN)")

    if unstructured is None:
        unstructured = x.ndim == 1

    # TODO: have this return GDF instead?
    # TODO: allowing specifying `crs`, `method`, shapely options (buffer, convex-hull), ...
    # TODO: optionally filter out unclosed?
    import matplotlib.pyplot as plt

    if unstructured:
        assert x.ndim == 1, "this is for a single time step"
        tri = kwargs.get("triangulation")
        if tri is None:
            from matplotlib.tri import Triangulation

            tri = Triangulation(x=x.lon, y=x.lat)
        with plt.ioff():  # requires mpl 3.4
            fig = plt.figure()
            cs = plt.tricontour(tri, x, levels=[value])
    else:
        assert x.ndim == 2, "this is for a single image"
        with plt.ioff():  # requires mpl 3.4
            fig = plt.figure()
            cs = x.plot.contour(x="lon", y="lat", levels=[value])  # type: ignore[attr-defined]

    plt.close(fig)
    assert len(cs.allsegs) == 1, "only one level"

    return cs.allsegs[0]


def _contours_to_gdf(cs: list[np.ndarray]) -> geopandas.GeoDataFrame:
    from geopandas import GeoDataFrame
    from shapely.geometry.polygon import LinearRing, LineString, Point, orient

    polys = []
    for c in cs:
        x, y = c.T
        if x.size < 3:
            logger.info(f"skipping an input contour with less than 3 points: {c}")
            continue
        r = LinearRing(zip(x, y))
        p0 = r.convex_hull  # TODO: optional, also add buffer option
        if type(p0) is LineString or type(p0) is Point:
            continue  # not a closed contour
        p = orient(p0)  # -> counter-clockwise
        polys.append(p)

    return GeoDataFrame(geometry=polys, crs="EPSG:4326")
    # ^ This crs indicates input in degrees


def _size_filter_contours(
    cs235: geopandas.GeoDataFrame,
    cs219: geopandas.GeoDataFrame,
    *,
    threshold: float = 4000,
) -> tuple[geopandas.GeoDataFrame, geopandas.GeoDataFrame]:
    """Compute areas and use to filter both sets of contours.

    `threshold` is for the total cold-core contour area within a given CE contour
    (units: km2).
    """
    import geopandas as gpd
    from shapely.geometry import MultiPolygon

    # Drop small 235s (a 235 with area < 4000 km2 can't have 219 area of 4000)
    cs235["area_km2"] = cs235.to_crs("EPSG:32663").area / 10**6
    # ^ This crs is equidistant cylindrical
    big_enough = cs235.area_km2 >= threshold
    if not big_enough.empty:
        logger.info(
            f"{big_enough.value_counts().get(True, 0) / big_enough.size * 100:.1f}% "
            f" of 235s are big enough ({threshold} km2)"
        )
    cs235 = cs235[big_enough].reset_index(drop=True)

    # Drop very small 219s (insignificant)
    # Note: This wasn't done in original TAMS, but here we are sometimes seeing
    # tiny 219 contours inside larger ones.
    individual_219_threshold = 10  # km2
    cs219["area_km2"] = cs219.to_crs("EPSG:32663").area / 10**6
    big_enough = cs219.area_km2 >= individual_219_threshold
    if not big_enough.empty:
        logger.info(
            f"{big_enough.value_counts().get(True, 0) / big_enough.size * 100:.1f}% "
            f" of 219s are big enough ({individual_219_threshold} km2)"
        )
    cs219 = cs219[big_enough].reset_index(drop=True)

    # Identify indices of 219s inside 235s
    # Note that some 235s might not have any 219s inside
    a = cs235.sjoin(cs219, predicate="contains", how="left").reset_index()
    # ^ gives an Int64 index with duplicated values, for each 219 inside a certain 235
    i219s = {  # convert to list
        i235: sorted(g.index_right.astype(int).to_list())
        for i235, g in a.groupby("index")
        if not g.index_right.isna().all()
    }

    # Store 219 info inside the 235 df
    cs235["inds219"] = [i219s.get(i235, []) for i235 in cs235.index]
    # TODO: possible to instead cleanly store the 219 `MultiPolygon`s in a column?
    # cs235["cs219"] = cs235.inds219.apply(lambda inds: cs219.iloc[inds][["geometry"]].dissolve().geometry)
    # cs235["cs219"] = cs235.inds219.apply(lambda inds: MultiPolygon(cs219.geometry.iloc[inds].values))

    # Check 219 area sum inside the 235 (total 219 area 4000 km2)
    sum219s = {
        i235: cs219.iloc[i219s.get(i235, [])].to_crs("EPSG:32663").area.sum() / 10**6
        for i235 in cs235.index
    }
    cs235["area219_km2"] = pd.Series(sum219s, dtype=float)
    big_enough = cs235.area219_km2 >= threshold
    if not big_enough.empty:
        logger.info(
            f"{big_enough.value_counts().get(True, 0) / big_enough.size * 100:.1f}% "
            f"of big-enough 235s have enough 219 area ({threshold} km2)"
        )
    cs235 = cs235[big_enough].reset_index(drop=True)

    # Store 219s inside the 235 frame as `MultiPolygon`s
    cs235["cs219"] = gpd.GeoSeries(
        cs235.inds219.apply(lambda inds: MultiPolygon(cs219.geometry.iloc[inds].values))
    )
    # NOTE: above method I found to be ~ 10x faster than this alternative:
    #   gpd.GeoSeries(cs235.inds219.apply(lambda inds: cs219.iloc[inds][["geometry"]].dissolve().geometry[0]))
    # The other benefit of applying `MultiPolygon` directly is that we get `MultiPolygon`
    # even in the case of only one, whereas dissolve gives `Polygon` in that case (row).
    #
    # TODO: alternative method dissolve on the 219s for each row like below leads to
    # slightly different data-in-contours pixel counts (greater?) in some cases
    #   f = lambda row: data_in_contours(tb, cs219.iloc[row.inds219].dissolve())
    #   res_ = cs235.apply(f, axis="columns")  # Series of DataFrames
    #   res = pd.concat(res_.to_list()).reset_index(drop=True)
    # TODO: unary_union (like dissolve uses), might be better unless can properly deal with
    # the small holes we sometimes get, otherwise some area gets counted extra

    # TODO: some elegant way to drop 219s that aren't inside a 235, resetting index
    # but preserving the matching of 235 to 219s

    return cs235, cs219


def _identify_one(
    ctt: xr.DataArray,
    *,
    size_filter: bool = True,
    ctt_threshold: float = 235,
    ctt_core_threshold: float = 219,
    unstructured: bool | None = None,
    triangulation: matplotlib.tri.Triangulation | None = None,
) -> tuple[geopandas.GeoDataFrame, geopandas.GeoDataFrame]:
    """Identify clouds in 2-D cloud-top temperature data `ctt` (e.g. at a specific time)."""

    cs235 = sort_ew(
        _contours_to_gdf(
            contours(ctt, ctt_threshold, unstructured=unstructured, triangulation=triangulation)
        )
    ).reset_index(drop=True)
    cs219 = sort_ew(
        _contours_to_gdf(
            contours(
                ctt, ctt_core_threshold, unstructured=unstructured, triangulation=triangulation
            )
        )
    ).reset_index(drop=True)

    if size_filter:
        cs235, cs219 = _size_filter_contours(cs235, cs219)

    return cs235, cs219


def identify(
    ctt: xarray.DataArray,
    *,
    size_filter: bool = True,
    parallel: bool = False,
    ctt_threshold: float = 235,
    ctt_core_threshold: float = 219,
) -> tuple[list[geopandas.GeoDataFrame], list[geopandas.GeoDataFrame]]:
    """Identify clouds in 2-D (lat/lon) or 3-D (lat/lon + time) cloud-top temperature data `ctt`.
    The 235 K contours returned (first list) serve to identify cloud elements (CEs).
    In a given frame from this list, each row corresponds to a certain CE.

    This is the first step in a TAMS workflow.

    Parameters
    ----------
    ctt
        Cloud-top temperature array.
    size_filter
        Whether to apply size-filtering
        (using 235 K and 219 K areas to filter out CEs that are not MCS material).
        Filtering at this stage makes TAMS more computationally efficient overall.
        Disable this option to return all identified CEs.
        Note that all 219s are returned regardless of this setting.

        When enabled, this also identifies the 219s (if any) that are within each 235.
    parallel
        Identify in parallel along ``'time'`` dimension for 3-D `ctt` (requires `joblib`).
    ctt_threshold
        Used to identify the edges of cloud elements.
    ctt_core_threshold
        Used to identify deep convective cloud regions within larger cloud areas.
        This is used to determine whether or not a system is eligible for being classified
        as an organized system.
        It helps target raining clouds.
    """
    dims = tuple(ctt.dims)

    unstructured = (len(dims) == 2 and "time" in dims) or (len(dims) == 1)

    triangulation = None
    if unstructured:
        if ctt.lat.ndim == 1 and ctt.lon.ndim == 1:
            from matplotlib.tri import Triangulation

            triangulation = Triangulation(x=ctt.lon, y=ctt.lat)
        else:
            warnings.warn(
                "detected unstructured data but not 1-D lat/lon "
                f"(got lat dims {ctt.lat.dims}, lon dims {ctt.lon.dims}), "
                "not pre-computing the triangulation"
            )

    f = functools.partial(
        _identify_one,
        size_filter=size_filter,
        ctt_threshold=ctt_threshold,
        ctt_core_threshold=ctt_core_threshold,
        unstructured=unstructured,
        triangulation=triangulation,
    )

    if (not unstructured and len(dims) == 2) or (unstructured and len(dims) == 1):
        cs235, cs219 = f(ctt)
        css235, css219 = (cs235,), (cs219,)  # to tuple for consistency

    elif "time" in dims and (
        (not unstructured and len(dims) == 3) or (unstructured and len(dims) == 2)
    ):
        assert ctt.time.ndim == 1
        itimes = np.arange(ctt.time.size)

        if parallel:
            try:
                import joblib
            except ImportError as e:
                raise RuntimeError("joblib required") from e

            res = joblib.Parallel(n_jobs=-2, verbose=10)(
                joblib.delayed(f)(ctt.isel(time=i)) for i in itimes
            )

        else:
            res = [f(ctt.isel(time=i)) for i in itimes]

        css235, css219 = zip(*res)

    else:
        raise ValueError(
            f"Got unexpected `ctt` dims: {dims}. "
            "They should be 2-D (lat/lon) + optional 'time' dim for structured grid data, "
            "or 1-D (cell) + optional 'time' dim for unstructured grid data."
        )

    return list(css235), list(css219)


def _data_in_contours_sjoin(
    data: xarray.DataArray | xarray.Dataset | pandas.DataFrame | geopandas.GeoDataFrame,
    contours: geopandas.GeoDataFrame,
    *,
    varnames: list[str],
    agg=("mean", "std", "count"),
) -> pandas.DataFrame:
    """Compute stats on `data` within `contours` using :func:`~geopandas.tools.sjoin`.

    `data` must have ``'lat'`` and ``'lon'`` variables.
    """
    import geopandas as gpd

    # Convert possibly-2-D data to GeoDataFrame of points
    if isinstance(data, gpd.GeoDataFrame):
        points = data
    else:
        if isinstance(data, pd.DataFrame):
            data_df = data
        else:
            data_df = data.to_dataframe().reset_index(drop=set(data.dims) != {"lat", "lon"})
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
    if not new_data_:
        raise ValueError("no data found in contours")
    new_data = pd.concat(new_data_).convert_dtypes()

    # Convert to standard (non-multi) index and str columns
    new_data = new_data.unstack()  # multi index -> (variable, agg) columns
    new_data.columns = ["_".join(s for s in tup) for tup in new_data.columns]

    return new_data


def _data_in_contours_regionmask(
    data: xarray.DataArray | xarray.Dataset,
    contours: geopandas.GeoDataFrame,
    *,
    varnames: list[str],
    agg=("mean", "std", "count"),
) -> pandas.DataFrame:
    import regionmask

    # Form regionmask(s)
    shapes = contours[["geometry"]]
    regions = regionmask.from_geopandas(shapes)
    mask = regions.mask(data)
    # Note: before Shapely v2, having `pygeos` installed made this faster

    # Aggregate points inside contour
    new_data_ = {
        i: data.where(mask == i).to_dataframe()[varnames].dropna().agg(agg).T
        for i in regions.numbers
    }
    if not new_data_:
        raise ValueError("no data found in contours")
    new_data = pd.concat(new_data_).convert_dtypes()
    # TODO: also try with xarray methods instead of going through pandas
    # TODO: try with xarray groupby

    # Convert to standard (non-multi) index and str columns
    new_data = new_data.unstack()  # multi index -> (variable, agg) columns
    new_data.columns = ["_".join(s for s in tup) for tup in new_data.columns]

    return new_data


def data_in_contours(
    data: xarray.DataArray | xarray.Dataset | pandas.DataFrame | geopandas.GeoDataFrame,
    contours: geopandas.GeoDataFrame,
    *,
    agg=("mean", "std", "count"),  # TODO: type
    method: str = "sjoin",
    merge: bool = False,
) -> pandas.DataFrame | geopandas.GeoDataFrame:
    """Compute statistics on `data` within the shapes of `contours`.

    With the default settings, we calculate,
    for each shape (row) in the `contours` dataframe:

    - the mean value of `data` within the shape
    - the standard deviation of `data` within the shape
    - the count of non-null values of `data` within the shape

    Parameters
    ----------
    data
        It should have ``'lat'`` and ``'lon'`` coordinates.
        If you pass a :class:`xarray.Dataset`,
        all :attr:`~xarray.Dataset.data_vars` will be included.
        If you pass a dataframe (supported for default `method` ``'sjoin'``),
        all columns except ``{'time', 'lat', 'lon', 'geometry'}`` will be included.
    contours
        For example, a dataframe of CE or MCS shapes, e.g.
        from :func:`identify` or :func:`track`.
    agg : sequence of str or callable
        Suitable for passing to :meth:`pandas.DataFrame.aggregate`.
    method : {'sjoin', 'regionmask'}
        The regionmask method is suited for data on a structured grid,
        while the GeoPandas sjoin method works for scattered point data as well.
        The sjoin method is the default since it is more general
        and currently often faster.
    merge
        Whether to merge the new data with `contours` or return a separate frame.
        If false (default), the index of the returned non-geo frame
        will be the same as that of `contours`
        (e.g. a row corresponding to an individual CE or MCS at a certain time).

    See Also
    --------
    :ref:`ctt_thresh_data_in_contours`
        A usage example.
    """
    import geopandas as gpd

    if isinstance(data, xr.DataArray):
        varnames = [data.name]
        if data.isnull().all():
            raise ValueError("Input array `data` is all null (e.g. NaN)")
            # TODO: warn instead for this and return cols of NaNs?
    elif isinstance(data, xr.Dataset):
        varnames = list(data.data_vars)
    elif isinstance(data, (pd.DataFrame, gpd.GeoDataFrame)):
        if method in {"regionmask"}:
            raise TypeError(f"method {method!r} requires `data` to be in xarray format")
        varnames = [vn for vn in data.columns if vn not in {"time", "lat", "lon", "geometry"}]
    else:
        raise TypeError(f"`data` has invalid type {type(data)!r}")

    if isinstance(agg, str):
        agg = (agg,)

    args = (data, contours)
    kwargs = dict(varnames=varnames, agg=agg)

    if method in {"sjoin", "geopandas", "gpd"}:
        new_data = _data_in_contours_sjoin(*args, **kwargs)
    elif method in {"regionmask"}:
        new_data = _data_in_contours_regionmask(*args, **kwargs)  # type: ignore[arg-type]
    else:
        raise ValueError(f"method {method!r} not recognized")

    if merge:
        # Merge with the `contours` gdf, appending columns
        new_data = contours.merge(new_data, left_index=True, right_index=True, how="left")

    return new_data


def _project_geometry(s: geopandas.GeoSeries, *, dx: float) -> geopandas.GeoSeries:
    crs0 = s.crs.to_string()

    return s.to_crs(crs="EPSG:32663").translate(xoff=dx).to_crs(crs0)


# TODO: test


def project(df: geopandas.GeoDataFrame, *, u: float = 0, dt: float = 3600):
    """Project the coordinates by `u` * `dt` meters in the *x* direction.

    Parameters
    ----------
    df
        Dataframe of objects to be spatially projected.
    u
        Speed [m s-1].
    dt
        Time [s]. Default: one hour.
    """
    dx = u * dt
    new_geometry = _project_geometry(df.geometry, dx=dx)

    return df.assign(geometry=new_geometry)


def overlap(a: geopandas.GeoDataFrame, b: geopandas.GeoDataFrame, *, norm: str = "a"):
    """For each contour in `a`, determine those in `b` that overlap and by how much.

    Currently the mapping is based on indices of the frames:
    iloc position in `a` : loc position in `b` : overlap fraction.

    Parameters
    ----------
    norm : {'a', 'b', 'max', 'min', 'mean'}
        Area to use to normalize the overlap to a fraction.
    """
    # TODO: test(s) for the different `norm` options
    s_crs_area = "EPSG:32663"
    area_a = a.to_crs(s_crs_area).area
    area_b = b.to_crs(s_crs_area).area

    res = {}
    for i in range(len(a)):
        a_i = a.iloc[i : i + 1]  # slicing preserves GeoDataFrame type
        a_i_poly = a_i.values[0][0]
        with warnings.catch_warnings():
            # We get this warning when an empty intersection is found
            warnings.filterwarnings(
                "ignore",
                category=RuntimeWarning,
                message="invalid value encountered in intersection",
            )
            inter = b.intersection(a_i_poly)
        inter = inter[~inter.is_empty]

        ov = inter.to_crs(s_crs_area).area

        if norm == "a":
            area_norm = area_a.iloc[i]
        elif norm == "b":
            area_norm = area_b.loc[ov.index]
        elif norm in {"max", "min", "mean"}:
            op = norm
            b_area_i = area_b.loc[ov.index]
            area_norm = getattr(
                pd.concat(
                    [
                        pd.Series(
                            data=np.full(len(b_area_i), area_a.iloc[i]),
                            index=b_area_i.index,
                        ),
                        b_area_i,
                    ],
                    axis="columns",
                ),
                op,
            )(axis="columns")
        else:
            raise ValueError(f"invalid `norm` {norm!r}")

        ov_frac = ov / area_norm

        res[i] = ov_frac.to_dict()

    return res


def track(
    contours_sets: list[geopandas.GeoDataFrame],
    times,  # TODO: could replace these two with single dict?
    *,
    overlap_threshold: float = 0.5,
    u_projection: float = 0,
    durations=None,
    look: str = "back",
    largest: bool = False,
    overlap_norm: str | None = None,
) -> geopandas.GeoDataFrame:
    """Assign group IDs to the CEs identified at each time, returning a single CE frame.

    Currently this works by: for each CE at the current time step,
    searching for a "parent" from the previous time step by computing
    overlap with all previous CEs.

    Parameters
    ----------
    contours_sets
        List of identified contours, in GeoDataFrame format.
    times
        Timestamps associated with each identified set of contours.
    overlap_threshold
        In [0, 1] (i.e., fractional), the overlap threshold.
    u_projection
        Zonal projection velocity, to project previous time step CEs by before
        computing overlap.
        5--13 m/s are typical magnitudes to use.
        For AEWs, a negative value should be used.
    durations
        Durations associated with the times in `times` (akin to the time resolution).
        If not provided, they will be estimated using ``times[1:] - times[:-1]``.
    look
        (time) direction in which we "look" and compute overlaps,
        linking CEs in time.
    largest
        Only the largest CE continues a track.
    overlap_norm
        Passed to :func:`overlap`.
        Default is to use child area (
        ``'a'`` for ``look='back'``,
        ``'b'`` for ``look='forward'``
        ), i.e. the CE at the later of the two times.
    """
    assert len(contours_sets) == len(times) and len(times) > 1
    times = pd.DatetimeIndex(times)
    itimes = list(range(times.size))

    if durations is not None:
        assert len(durations) == len(times)
    else:
        # Estimate dt values
        dt = times[1:] - times[:-1]
        assert (dt.astype(np.int64) > 0).all()
        if not dt.unique().size == 1:
            warnings.warn("unequal time spacing")
        dt = dt.insert(-1, dt[-1])

    # IDEA: even at initial time, could put CEs together in groups based on edge-to-edge distance

    if look in {"f", "forward"}:
        warnings.warn("forward `look` considered experimental")

    css: list[geopandas.GeoDataFrame] = []
    for i in itimes:
        cs_i = contours_sets[i]
        cs_i["time"] = times[i]  # actual time
        cs_i["itime"] = itimes[i]  # time index (from 0)
        cs_i["dtime"] = dt[i]  # delta time
        n_i = len(cs_i)
        if i == 0:
            # IDs all new for first time step
            cs_i["mcs_id"] = range(n_i)
            next_id = n_i
        else:
            # Assign IDs using overlap threshold
            cs_im1 = css[i - 1]
            dt_im1_s = dt[i - 1].total_seconds()

            if look in {"b", "back"}:
                if overlap_norm is None:
                    overlap_norm = "a"
                ovs = overlap(cs_i, project(cs_im1, u=u_projection, dt=dt_im1_s), norm=overlap_norm)
                ids = []
                for j, d in ovs.items():
                    # For each CE at current time ("kid"/"child"),
                    # find previous CE of maximum overlap (single "parent")
                    k, frac = max(d.items(), key=lambda tup: tup[1], default=(None, 0))
                    if k is None or frac < overlap_threshold:
                        # No parent or not enough overlap => new ID
                        ids.append(next_id)
                        next_id += 1
                    else:
                        # Has parent; use their family ID
                        ids.append(cs_im1.loc[k].mcs_id)

                assert len(ids) == len(cs_i)

                if largest:
                    # For current CEs, make sure no MCS ID is shared (give to largest)
                    sz = cs_i["area_km2"].to_numpy(dtype=np.float64)
                    for mcs_id in set(ids):
                        inds_with_id = [j for j, id_ in enumerate(ids) if id_ == mcs_id]
                        if len(inds_with_id) == 1:
                            continue
                        logger.info(f"multiple CEs with same MCS ID: {inds_with_id}")
                        ind_largest, _ = max(
                            zip(inds_with_id, sz[inds_with_id]),
                            key=lambda tup: tup[1],
                        )
                        logger.info(f"largest: {sz[ind_largest]} out of {sz[inds_with_id]}")
                        for j in inds_with_id:
                            if j == ind_largest:
                                continue
                            ids[j] = next_id
                            next_id += 1
                        assert ids[ind_largest] == mcs_id

            elif look in {"f", "forward"}:
                # Following `look='b'`,
                # `j`: iloc of CE at current time (i, `cs_i`)
                # `k`: iloc of CE at previous time (i-1, `cs_im1`)
                if overlap_norm is None:
                    overlap_norm = "b"
                ovs = overlap(project(cs_im1, u=u_projection, dt=dt_im1_s), cs_i, norm=overlap_norm)
                ids: list[int | None] = [None for _ in range(len(cs_i))]  # type: ignore[no-redef]
                sz_i = cs_i["area_km2"].to_numpy(dtype=np.float64)
                for k, d in ovs.items():
                    # For each CE at previous time ("parent"),
                    # look at CEs of current time that overlap ("kid"/"child")
                    mcs_id = cs_im1.loc[k].mcs_id
                    if not d:
                        continue

                    if largest and len(d) > 1:
                        # NOTE: 1-1 track not guaranteed since multiple parents could have same ID
                        # j, frac = max(d.items(), key=lambda tup: tup[1])  # max overlap
                        js = list(d.keys())
                        logger.info(f"{len(d)} possible children: {js}")
                        sz_ji = sz_i[js]
                        j, frac, _ = max(
                            zip(d.keys(), d.values(), sz_ji),
                            key=lambda tup: tup[2],
                        )  # max child area
                        assert sz_i[j] == sz_ji.max()
                        logger.info(f"keeping largest: {sz_i[j]} out of {sz_ji}")
                        d = {j: frac}

                    for j, frac in d.items():
                        if frac >= overlap_threshold:
                            if ids[j] is not None:
                                logger.info(
                                    f"warning: {j} already set to ID {ids[j]}, now {mcs_id}"
                                )
                                # TODO: support multiple parent
                            # Assign child the MCS ID of parent
                            ids[j] = mcs_id

                # For current CEs with no assigned parent, give new IDs
                for j, mcs_id in enumerate(ids):
                    if mcs_id is None:
                        ids[j] = next_id
                        next_id += 1

            else:
                raise ValueError("invalid `look`")

            cs_i["mcs_id"] = ids

        css.append(cs_i)

    # Combine into one frame
    cs = pd.concat(css)

    return cs.reset_index(drop=True)  # drop nested time, CE ind index


def calc_ellipse_eccen(p: shapely.geometry.polygon.Polygon):
    """Compute the (first) eccentricity of the least-squares best-fit ellipse
    to the coordinates of the polygon's exterior.
    """
    # TODO: Ellipse class with methods to convert to shapely Polygon/LinearRing and mpl Ellipse Patch

    # using skimage https://scikit-image.org/docs/dev/api/skimage.measure.html#skimage.measure.EllipseModel
    from skimage.measure import EllipseModel

    xy = np.asarray(p.exterior.coords)
    assert xy.shape[1] == 2

    m = EllipseModel()
    success = m.estimate(xy)

    if not success:
        warnings.warn(f"ellipse model failed for {p}")
        return np.nan

    _, _, xhw, yhw, _ = m.params
    # ^ xc, yc, a, b, theta; from the docs
    #   a with x, b with y (after subtracting the rotation), but they are half-widths
    #   theta is in radians

    rat = yhw / xhw if xhw > yhw else xhw / yhw

    return np.sqrt(1 - rat**2)


def _classify_one(cs: geopandas.GeoDataFrame) -> str:
    """Classify one CE family group."""
    # eps = sqrt(1 - (b^2/a^2)) -- ellipse "first eccentricity"
    #
    # Below from most to least strict:
    #
    # MCCs (organized)
    # - 219 K region >= 25k km2
    # - 235 K region >= 50k km2
    # - size durations have to be met for >= 6 hours
    # - eps <= 0.7
    #
    # CCCs (organized)
    # - 219 K region >= 25k km2
    # - size durations have to be met for >= 6 hours
    # - no shape criterion
    #
    # DLL (disorganized)
    # - >= 6 hour duration
    # - (no size or shape criterion)
    #
    # DSL (disorganized)
    # - < 6 hour duration
    #
    # Classification is for the "family" groups

    assert cs.mcs_id.unique().size == 1, "this is for a certain CE family group"

    # Sum areas over cloud elements
    time_groups = cs.groupby("time")
    area = time_groups[["area_km2", "area219_km2"]].apply(sum)

    # Get duration (time resolution of our CE data)
    dt = time_groups["dtime"].apply(_the_unique)
    dur_tot = dt.sum()

    # Compute area-duration criteria
    dur_219_25k = dt[area.area219_km2 >= 25_000].sum()
    dur_235_50k = dt[area.area_km2 >= 50_000].sum()
    six_hours = pd.Timedelta(hours=6)

    if dur_219_25k >= six_hours:  # organized
        # Compute ellipse eccentricity
        eps = time_groups[["geometry"]].apply(
            lambda g: calc_ellipse_eccen(g.dissolve().geometry.convex_hull.iloc[0])
        )
        dur_eps = dt[eps <= 0.7].sum()
        if dur_235_50k >= six_hours and dur_eps >= six_hours:
            class_ = "MCC"
        else:
            class_ = "CCC"

    else:  # disorganized
        if dur_tot >= six_hours:
            class_ = "DLL"
        else:
            class_ = "DSL"

    return class_


def classify(cs: geopandas.GeoDataFrame) -> geopandas.GeoDataFrame:
    """Classify the CE groups into MCS classes, adding a categorical ``'mcs_class'`` column
    to the input frame.
    """

    assert {"mcs_id", "time", "dtime"} < set(cs.columns), "needed by the classify algo"

    classes = cs.groupby("mcs_id").apply(_classify_one)
    cs["mcs_class"] = cs.mcs_id.map(classes).astype("category")

    return cs


def run(
    ds: xarray.DataArray,
    *,
    parallel: bool = True,
    u_projection: float = 0,
    ctt_threshold: float = 235,
    ctt_core_threshold: float = 219,
) -> tuple[geopandas.GeoDataFrame, geopandas.GeoDataFrame, geopandas.GeoDataFrame]:
    r"""Run all TAMS steps, including precip assignment.

    .. important::
       `ds` must have ``'ctt'`` (cloud-top temperature) and ``'pr'`` (precip rate) variables.
       Dims should be ``'time'``, ``'lat'``, ``'lon'``.
       ``'lon'`` should be in -180 -- 180 format.


    Usage:

    >>> ce, mcs, mcs_summary = tams.run(ds)

    Parameters
    ----------
    ds
        Dataset containing 3-D cloud-top temperature and precipitation rate.
    parallel
        Whether to apply parallelization (where possible).
    u_projection
        *x*\-direction projection velocity to apply before computing overlaps.
    ctt_threshold
        Used to identify the edges of cloud elements.
    ctt_core_threshold
        Used to identify deep convective cloud regions within larger cloud areas.
        This is used to determine whether or not a system is eligible for being classified
        as an organized system.
        It helps target raining clouds.

    See Also
    --------
    :func:`tams.identify`
    :func:`tams.track`
    :func:`tams.classify`

    :doc:`/examples/tams-run`
    """
    import itertools

    import geopandas as gpd
    from shapely.errors import ShapelyDeprecationWarning
    from shapely.geometry import MultiPolygon

    assert {"ctt", "pr"} <= set(ds.data_vars)
    assert "time" in ds.dims

    # TODO: timing and progress indicators, possibly with Rich

    def msg(s):
        """Print message and current time"""
        import datetime

        st = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(s, st)

    #
    # 1. Identify
    #

    msg("Starting `identify`")
    cs235, cs219 = identify(
        ds.ctt,
        parallel=parallel,
        ctt_threshold=ctt_threshold,
        ctt_core_threshold=ctt_core_threshold,
    )

    #
    # 2. Track
    #

    msg("Starting `track`")
    times = ds.time.values
    dt = pd.Timedelta(times[1] - times[0])  # TODO: equal spacing check here?
    ce = track(cs235, times, u_projection=u_projection)

    #
    # 3. Classify
    #

    msg("Starting `classify`")
    ce = classify(ce)

    #
    # 4. Stats (including precip)
    #

    msg("Starting statistics calculations")

    # Cleanup
    ce = ce.drop(columns=["inds219", "itime", "dtime"]).convert_dtypes()
    ce = ce.rename(columns={"geometry": "cs235"}).set_geometry("cs235")
    ce.cs219 = ce.cs219.set_crs("EPSG:4326")  # TODO: ensure set in `identify`

    msg("Starting CE aggregation (into MCS time series)")
    dfs_t = []
    ds_nt = []
    for mcs_id, mcs_ in ce.groupby("mcs_id"):
        # Time-varying
        time_group = mcs_.groupby("time")
        d = {}

        with warnings.catch_warnings():
            # ShapelyDeprecationWarning: __len__ for multi-part geometries is deprecated ...
            warnings.filterwarnings(
                "ignore",
                category=ShapelyDeprecationWarning,
                message="__len__ for multi-part geometries is deprecated",
            )
            d["cs235"] = gpd.GeoSeries(time_group.apply(lambda g: MultiPolygon(g.geometry.values)))
            d["cs219"] = gpd.GeoSeries(
                time_group.apply(
                    lambda g: MultiPolygon(
                        itertools.chain.from_iterable(mp.geoms for mp in g.cs219.values)
                    )
                )
            )

        d["nce"] = len(mcs_)
        d["area_km2"] = time_group.area_km2.sum()
        d["area219_km2"] = time_group.area219_km2.sum()
        # TODO: compare to re-computing area after (could be different if shift to dissolve)?

        df = pd.DataFrame(d).reset_index()  # time -> column
        df["mcs_id"] = mcs_id
        assert mcs_.mcs_class.unique().size == 1
        df["mcs_class"] = mcs_.mcs_class.values[0]

        # Summary stuff
        d2 = {}
        times = mcs_.time.unique()
        d2["first_time"] = times.min()
        d2["last_time"] = times.max()
        d2["duration"] = d2["last_time"] - d2["first_time"] + dt
        d2["mcs_id"] = mcs_id
        d2["mcs_class"] = mcs_.mcs_class.values[0]

        dfs_t.append(df)
        ds_nt.append(d2)

    # Initial MCS time-resolved
    mcs = (
        gpd.GeoDataFrame(pd.concat(dfs_t).reset_index(drop=True))
        .set_geometry("cs235", crs="EPSG:4326")
        .convert_dtypes()
    )
    mcs.cs219 = mcs.cs219.set_crs("EPSG:4326")
    mcs.mcs_class = mcs.mcs_class.astype("category")

    # Add CTT and PR data stats (time-resolved)
    msg("Starting gridded data aggregation")

    def _agg_one(ds_t, g):
        df1 = data_in_contours(ds_t.pr, g, merge=True)
        df2 = data_in_contours(ds_t.pr, g.set_geometry("cs219", drop=True), merge=False).add_suffix(
            "219"
        )
        df3 = data_in_contours(
            ds_t.ctt, g.set_geometry("cs219", drop=True), merge=False
        ).add_suffix("219")
        df = (
            df1.join(df2)
            .join(df3)
            .drop(
                columns=[
                    "count_pr219",
                ]
            )
            .rename(columns={"count_pr": "npixel", "count_ctt219": "npixel219"})
        )
        return df

    if parallel:
        try:
            import joblib
        except ImportError as e:
            raise RuntimeError("joblib required") from e

        # TODO: Sometimes getting
        # > UserWarning: A worker stopped while some jobs were given to the executor.
        # > This can be caused by a too short worker timeout or by a memory leak.
        # Increasing `batch_size` reduces the number of these (e.g. to 1 with batch 10, 119 jobs, 11 workers).
        # Probably better to leave on auto to keep more general though.
        # Run time doesn't seem affected.
        dfs = joblib.Parallel(n_jobs=-2, verbose=10, batch_size="auto")(
            joblib.delayed(_agg_one)(ds.sel(time=t).copy(deep=True), g.copy())
            for t, g in mcs.groupby("time")
        )
    else:
        dfs = [_agg_one(ds.sel(time=t), g) for t, g in mcs.groupby("time")]

    mcs = pd.concat(dfs)

    # Initial MCS summary
    mcs_summary = pd.DataFrame(ds_nt).reset_index(drop=True).convert_dtypes()
    mcs_summary.mcs_class = mcs_summary.mcs_class.astype("category")

    # Add some CTT and PR stats to summary dataset
    msg("Computing stats for MCS summary dataset")
    vns = [
        "mean_pr",
        "mean_pr219",
        "mean_ctt219",
        "std_ctt219",
        "area_km2",
        "area219_km2",
    ]
    mcs_summary = mcs_summary.join(
        mcs.groupby("mcs_id")[vns].mean().rename(columns={vn: f"mean_{vn}" for vn in vns})
    )

    # Add first and last points and distance to MCS summary dataset,
    # setting first point as the `geometry`

    def f(g):
        g.sort_values(by="time")  # should be already but just in case...
        cen = g.geometry.to_crs("EPSG:32663").centroid.to_crs("EPSG:4326")
        return gpd.GeoSeries({"first_centroid": cen.iloc[0], "last_centroid": cen.iloc[-1]})

    mcs_summary_points = gpd.GeoDataFrame(mcs.groupby("mcs_id").apply(f).astype("geometry"))
    # ^ Initially we have GeoDataFrame but the columns don't have dtype geometry
    # `.astype("geometry")` makes that conversion but we lose GeoDataFrame

    # `.set_crs()` only works on a dtype=geometry column in a GeoDataFrame
    mcs_summary_points.first_centroid = mcs_summary_points.first_centroid.set_crs("EPSG:4326")
    mcs_summary_points.last_centroid = mcs_summary_points.last_centroid.set_crs("EPSG:4326")
    assert (
        mcs_summary_points.first_centroid.crs == mcs_summary_points.last_centroid.crs == "EPSG:4326"
    )

    mcs_summary_points["distance_km"] = (
        mcs_summary_points.first_centroid.to_crs("EPSG:32663").distance(
            mcs_summary_points.last_centroid.to_crs("EPSG:32663")
        )
        / 10**3
    ).astype("Float64")

    mcs_summary = (
        gpd.GeoDataFrame(mcs_summary).join(mcs_summary_points).set_geometry("first_centroid")
    )

    # Final cleanup
    mcs = mcs.reset_index(drop=True)
    mcs_summary = mcs_summary.reset_index(drop=True)

    msg("Done")

    return ce, mcs, mcs_summary


if __name__ == "__main__":
    import cartopy.crs as ccrs
    import matplotlib.pyplot as plt
    import regionmask

    from .data import load_example_ir, tb_from_ir

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
    mask = regions.mask(tb)

    regions.plot(ax=ax)

    # tb.where(mask >= 0).plot.pcolormesh(ax=ax, transform=tran)  # takes long
    tb.where(mask >= 0).plot.pcolormesh(size=4, aspect=2)

    plt.show()
