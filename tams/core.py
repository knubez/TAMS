"""
Core routines that make up the TAMS algorithm.

Functions for external use should be exported from the top-level;
users shouldn't need to import/use objects from this module directly.
"""

from __future__ import annotations

import functools
import warnings
from typing import TYPE_CHECKING, NamedTuple

import numpy as np
import pandas as pd
import xarray as xr

from .util import _the_unique, get_logger, get_worker_logger, sort_ew

if TYPE_CHECKING:
    import geopandas
    import numpy
    import pandas
    import shapely
    import xarray
    from matplotlib.tri import Triangulation


logger = get_logger()


def _contour_segs_to_gdf(
    segs: list[numpy.ndarray],
    *,
    closed_only: bool = True,
    tolerance: float = 1e-12,
) -> geopandas.GeoDataFrame:
    """Convert contour segments from matplotlib to a GeoDataFrame.

    Parameters
    ----------
    segs
        List of 2-D arrays describing contours.
        The arrays are shape ``(n, 2)``; each row is a coordinate pair.
    closed_only
        Only return closed contours.
    tolerance
        Tolerance for snapping the start and end points of a contour together
        and for removing repeated points.

    Returns
    -------
    :
        GeoDataframe of analyzed contours.
        Closed contours as :class:`~shapely.LinearRing`,
        open contours as :class:`~shapely.LineString`.
    """
    from collections import Counter

    import geopandas as gpd
    from shapely import LinearRing, LineString, Polygon, remove_repeated_points
    from shapely.errors import TopologicalError
    from shapely.geometry.polygon import orient
    from shapely.validation import explain_validity

    logger = get_worker_logger()

    if not tolerance >= 0:
        raise ValueError("tolerance must be non-negative")

    n0 = len(segs)
    skipped: Counter[str] = Counter()
    logger.info(f"processing {n0} contours")
    data = []
    for c in segs:
        if c.shape[0] < 2:
            # 0 -> empty LinearRing or LineString
            # 1 -> LineString GEOS exception (needs 2)
            # 1-2 -> LinearRing ValueError (needs 4)
            # 3 -> LinearRing will add the first point to the end to make 4
            skipped["too few points"] += 1
            continue

        # Snap closed if close
        if np.isclose(c[0], c[-1], atol=tolerance, rtol=0).all():
            c[-1] = c[0]

        is_closed = (c[-1] == c[0]).all()
        ls = remove_repeated_points(LineString(c), tolerance=tolerance)

        # "closed line loops are oriented anticlockwise
        # if they enclose a region that is higher then the contour level,
        # or clockwise if they enclose a region that is lower than the contour level"
        if is_closed:
            try:
                r = LinearRing(ls)
            except (ValueError, TopologicalError) as e:
                skipped[f"invalid closed ({e})"] += 1
                continue
            if not r.is_valid:
                skipped[f"invalid closed ({explain_validity(r)})"] += 1
                continue
            encloses_higher = r.is_ccw
            r_ccw = orient(Polygon(r)).exterior  # ensure consistent
            assert r_ccw.is_valid
            data.append(
                {
                    "contour": r_ccw,
                    "closed": is_closed,
                    "encloses_higher": encloses_higher,
                }
            )
        elif closed_only:
            skipped["open"] += 1
            continue
        else:
            encloses_higher = None
            if not ls.is_valid:
                skipped[f"invalid open ({explain_validity(ls)})"] += 1
                continue
            if not ls.is_simple:
                # e.g. self-intersecting
                skipped["non-simple open"] += 1
                continue
            data.append(
                {
                    "contour": ls,
                    "closed": is_closed,
                    "encloses_higher": encloses_higher,
                }
            )

    assert skipped.total() == n0 - len(data)
    if skipped:
        logger.debug(
            "skipped contours: " + ", ".join(f"{k}: {v}" for k, v in sorted(skipped.items()))
        )
    logger.info(f"returning {len(data)}/{n0} contours")
    if not data:
        data = {  # type: ignore[assignment]
            "contour": [],
            "closed": [],
            "encloses_higher": [],
        }

    return gpd.GeoDataFrame(
        data,
        geometry="contour",
        crs="EPSG:4326",
    ).astype(
        {
            "closed": bool,
            "encloses_higher": "boolean",  # nullable
        },
    )


def contour(
    x: xarray.DataArray,
    value: float,
    *,
    unstructured: bool | None = None,
    triangulation: Triangulation | None = None,
    closed_only: bool = True,
    tolerance: float = 1e-12,
) -> geopandas.GeoDataFrame:
    """Contour `x` at `value`, producing a dataframe of contour lines.

    Parameters
    ----------
    x
        Data to be contoured.
        Needs to have ``'lat'`` and ``'lon'`` coordinates.
        The array should be 2-D if structured, 1-D if unstructured.
    value
        Find contours where `x` has this value.
    unstructured
        Whether the grid of `x` is unstructured (e.g. MPAS native output).
        Default: assume unstructured if `x` is 1-D.
    triangulation : Triangulation, optional
        A pre-computed :class:`~matplotlib.tri.Triangulation` for `x`
        with unstructured grid.
    closed_only
        Only return closed contours.
    tolerance
        Tolerance for snapping the start and end points of a contour together
        and for removing repeated points.

    Returns
    -------
    cs : GeoDataFrame
        GeoDataframe of analyzed contours.
        Closed contours as :class:`~shapely.LinearRing`,
        open contours as :class:`~shapely.LineString`.

        Columns:

        - ``contour`` -- geometry, the contours
        - ``closed`` -- bool, whether the contour is closed
        - ``encloses_higher`` -- nullable bool, if closed, whether the contour
          encloses a region with values higher than `value`.
          If not closed, this is null.

    Raises
    ------
    ValueError
        If all values in `x` are null.
        Or if `x` has an unexpected number of dimensions.
    """
    import matplotlib.pyplot as plt

    logger = get_worker_logger()

    if x.isnull().all():
        raise ValueError("input array `x` is all null (e.g. NaN)")

    if unstructured is None:
        unstructured = x.ndim == 1

    name = x.name or "?"
    s_dims = ", ".join(f"{d}: {n}" for d, n in x.sizes.items())
    s_tri = "..." if triangulation is not None else "None"
    logger.info(
        f"contouring {name} ({s_dims}) at {value}, "
        f"unstructured={unstructured}, triangulation={s_tri}, "
        f"closed_only={closed_only}, tolerance={tolerance}"
    )

    if unstructured:
        if not x.ndim == 1:
            raise ValueError(
                "this is for a single time step. For unstructured grid there should be one dim."
            )
        tri = triangulation
        if tri is None:
            from matplotlib.tri import Triangulation

            tri = Triangulation(x=x.lon, y=x.lat)
        with plt.ioff():  # requires mpl 3.4
            fig = plt.figure()
            cs = plt.tricontour(tri, x, levels=[value])
    else:
        if not x.ndim == 2:
            raise ValueError("this is for a single image")
        with plt.ioff():  # requires mpl 3.4
            fig = plt.figure()
            cs = x.plot.contour(x="lon", y="lat", levels=[value])

    plt.close(fig)
    assert len(cs.allsegs) == 1, "only one level"

    return _contour_segs_to_gdf(cs.allsegs[0], closed_only=closed_only, tolerance=tolerance)


def _contours_to_polygons(
    cs: geopandas.GeoDataFrame,
    *,
    edge_encloses_higher: bool | None = None,
) -> geopandas.GeoDataFrame:
    """Take the closed contours in `cs` (from :func:`contour`),
    which are :class:`~shapely.LineString`,
    and convert to :class:`~shapely.Polygon`, accounting for holes."""
    from collections import defaultdict

    import geopandas as gpd
    from shapely import Polygon
    from shapely.geometry.polygon import orient

    if cs.empty:
        return gpd.GeoDataFrame(
            geometry=[],
            crs="EPSG:4326",
        )

    # Preprocess by selecting closed contours, converting to polygons,
    # computing area, and sorting (smallest -> largest)
    cs = (
        cs.query("closed")
        .assign(contour=cs.geometry.apply(lambda r: Polygon(r)))
        .assign(area_km2=lambda df: df.geometry.to_crs("EPSG:32663").area / 1e6)
        .sort_values("area_km2")
        .reset_index(drop=True)
    )
    if edge_encloses_higher is None:
        largest = cs.iloc[-1]
        edge_encloses_higher = largest.encloses_higher

    contains = cs.sjoin(cs, predicate="contains", how="left")

    # Work from smallest outwards
    # A contour is a hole if it is within another contour and its `encloses_higher`
    # is opposite that of `edge_encloses_higher`.
    # Take the smallest container to be the parent.
    # A hole can only belong to one parent,
    # though a parent can have multiple holes.
    hole_inds = defaultdict(list)
    all_holes = set()
    for _, g in contains.rename_axis("index_left").reset_index().groupby("index_left"):
        for _, row in g.iterrows():
            if (
                row.encloses_higher_right != edge_encloses_higher
                and row.encloses_higher_left == edge_encloses_higher
                and row.index_right not in all_holes
            ):
                hole_inds[row.index_left].append(row.index_right)
                all_holes.add(row.index_right)

    # Construct new polygons from the existing
    new_polys = []
    for i in cs.index:
        if i in hole_inds:
            p = Polygon(
                cs.loc[i].contour.exterior,
                [orient(cs.loc[j].contour, -1).exterior for j in hole_inds[i]],
            )
            new_polys.append(p)
        elif i in all_holes:
            continue
        else:
            # Technically, an outermost edge could still be a hole
            if cs.loc[i].encloses_higher != edge_encloses_higher:
                continue
            new_polys.append(cs.loc[i].contour)

    return gpd.GeoDataFrame(
        geometry=new_polys,
        crs="EPSG:4326",
    )


def _size_filter(
    ce: geopandas.GeoDataFrame,
    core: geopandas.GeoDataFrame,
    *,
    threshold: float = 4000,
) -> geopandas.GeoDataFrame:
    """Compute areas, associate core areas with cloud elements,
    filter based on size threshold,
    returning the CE frame only.

    `threshold` is for the total cold-core contour area within a given CE contour
    (units: km2).
    """
    import geopandas as gpd

    logger = get_worker_logger()

    # Drop small CEs (a CE with area < 4000 km2 can't have cold-core area of 4000)
    ce["area_km2"] = ce.to_crs("EPSG:32663").area / 10**6
    # ^ This crs is equidistant cylindrical
    big_enough = ce.area_km2 >= threshold
    if not big_enough.empty:
        logger.info(
            f"{big_enough.value_counts().get(True, 0) / big_enough.size * 100:.1f}% "
            f"of CEs are big enough ({threshold} km2)"
        )
    ce = ce[big_enough].reset_index(drop=True)

    # Drop very small cold cores (insignificant)
    # Note: This wasn't done in original TAMS, but here we are sometimes seeing
    # tiny core contours inside larger ones.
    individual_core_threshold = min(threshold, 10)  # km2
    core["area_km2"] = core.to_crs("EPSG:32663").area / 10**6
    big_enough = core.area_km2 >= individual_core_threshold
    if not big_enough.empty:
        logger.info(
            f"{big_enough.value_counts().get(True, 0) / big_enough.size * 100:.1f}% "
            f"of cores are big enough ({individual_core_threshold} km2)"
        )
    core = core[big_enough].reset_index(drop=True)

    # Identify the cold cores that are inside CEs and store them inside the CE frame
    # Note that some CEs might not have any cores inside (-> None),
    # and while some have one (Polygon), some may have more than one (-> MultiPolygon)
    ce_core_as_index = ce.index.map(
        core.sjoin(
            ce,
            predicate="within",
            how="inner",
            lsuffix="core",
            rsuffix="ce",
        )
        .groupby("index_ce")
        .geometry.apply(lambda gs: gs.union_all(method="unary"))
    )
    ce["core"] = gpd.GeoSeries(
        # When empty, dtype is int64
        ce_core_as_index.astype("geometry"),
        crs="EPSG:4326",
    )

    # Drop CEs whose total cold-core area doesn't meet the threshold
    ce["area_core_km2"] = (ce.core.to_crs("EPSG:32663").area / 10**6).fillna(0)
    big_enough = ce.area_core_km2 >= threshold
    if not big_enough.empty:
        logger.info(
            f"{big_enough.value_counts().get(True, 0) / big_enough.size * 100:.1f}% "
            f"of big-enough CEs have enough cold-core area ({threshold} km2)"
        )
    ce = ce[big_enough].reset_index(drop=True)

    return ce


def _identify_one(
    ctt: xr.DataArray,
    *,
    ctt_threshold: float = 235,
    ctt_core_threshold: float = 219,
    size_filter: bool = True,
    size_threshold: float = 4000,
    convex_hull: bool = True,
    unstructured: bool | None = None,
    triangulation: Triangulation | None = None,
) -> geopandas.GeoDataFrame:
    """Identify clouds in 2-D cloud-top temperature data `ctt` (e.g. at a specific time)."""

    logger = get_worker_logger()

    name = ctt.name or "?"
    time = ctt.coords.get("time", None)
    if time is None:
        s_time = "?"
    else:
        s_time = str(time.values[()])
    logger.info(f"identifying CEs in {name} at time {s_time}")

    kws = dict(
        unstructured=unstructured,
        triangulation=triangulation,
        closed_only=True,
    )
    ce = (
        contour(ctt, ctt_threshold, **kws)  # type: ignore[arg-type]
        .pipe(_contours_to_polygons)
        .pipe(sort_ew)
        .reset_index(drop=True)
    )
    core = (
        contour(ctt, ctt_core_threshold, **kws)  # type: ignore[arg-type]
        .pipe(_contours_to_polygons)
        .pipe(sort_ew)
        .reset_index(drop=True)
    )

    if convex_hull:
        ce["geometry"] = ce.geometry.convex_hull
        core["geometry"] = core.geometry.convex_hull

    if not size_filter:
        warnings.warn(
            "Disabling size filtering via the `size_filter` argument is deprecated. "
            "Instead set `size_threshold=0` to achieve the same effect.",
            FutureWarning,
            stacklevel=3,
        )
        size_threshold = 0

    ce = _size_filter(ce, core, threshold=size_threshold)

    return ce


def identify(
    ctt: xarray.DataArray,
    *,
    ctt_threshold: float = 235,
    ctt_core_threshold: float = 219,
    size_filter: bool = True,
    size_threshold: float = 4000,
    convex_hull: bool = True,
    parallel: bool = False,
) -> list[geopandas.GeoDataFrame]:
    """Identify clouds in 2-D (lat/lon) or 3-D (lat/lon + time) cloud-top temperature data `ctt`.
    The returned list of polygon dataframes serves to identify cloud elements (CEs).
    In a given frame from this list, each row corresponds to a certain CE.

    This is the first step in a TAMS workflow.

    Parameters
    ----------
    ctt
        Cloud-top temperature array.
    ctt_threshold
        Used to identify the edges of cloud elements (CEs).
    ctt_core_threshold
        Used to identify deep convective cloud regions within larger cloud areas (cold cores).
        This is used to determine whether or not a system is eligible for being classified
        as an organized system.
        It helps target raining clouds.
    size_filter
        Whether to apply size-filtering
        (using CE and cold-core areas to filter out CEs that are not MCS material).
        Only CEs with enough cold-core area (`size_threshold`) are kept.

        .. deprecated:: 0.2.0
           Set ``size_threshold=0`` instead to disable size filtering.
    size_threshold
        Cold-core area threshold (units: km²).
        CEs with total cold-core area below this threshold
        are considered not MCS material and are filtered out.
        Set to 0 to disable size filtering (e.g. in order to do your own).
        Note that filtering at this stage makes TAMS more computationally efficient overall.
    convex_hull
        Apply convex hull to the CE polygons to simplify the shapes.

        .. note::

           * This is done before size filtering / area computation.
           * This fills in any holes the CE polygons may have.

        .. versionadded:: 0.2.0
           In v0.1.x it was not possible to disable convex hulling.
    parallel
        Identify in parallel along ``'time'`` dimension for 3-D `ctt` (requires `joblib`).

    Returns
    -------
    ces : list of GeoDataFrame
        List of dataframes of CE polygons.
        Columns:

        - ``geometry`` -- geometry, the CE polygons
        - ``area_km2`` -- float, area of the CE polygons (km²)
        - ``core`` -- geometry, the cold cores within each CE
          (:class:`~shapely.MultiPolygon`, :class:`~shapely.Polygon`, or ``None`` if no cores)
        - ``area_core_km2`` -- float, the CE's cold-core area (km²)

    See Also
    --------
    :doc:`/examples/identify`
        Demonstrating the impacts of options.

    :func:`contour`
        A lower-level and more general routine for producing shapes by contouring a threshold.
    """
    # TODO: allowing specifying `crs`, `method`, shapely options (buffer, convex-hull), ...
    dims = tuple(ctt.dims)

    s_dims = ", ".join(f"{d}: {n}" for d, n in ctt.sizes.items())
    name = ctt.name or "?"
    logger.info(
        f"identifying CEs in {name} ({s_dims}), "
        f"ctt_threshold={ctt_threshold}, ctt_core_threshold={ctt_core_threshold}, "
        f"size_filter={size_filter}, size_threshold={size_threshold}, "
        f"convex_hull={convex_hull}, "
        f"parallel={parallel}"
    )

    unstructured = (len(dims) == 2 and "time" in dims) or (len(dims) == 1)
    logger.debug(f"assuming unstructured={unstructured}")

    triangulation = None
    if unstructured:
        if ctt.lat.ndim == 1 and ctt.lon.ndim == 1:
            from matplotlib.tri import Triangulation

            triangulation = Triangulation(x=ctt.lon, y=ctt.lat)
        else:
            warnings.warn(
                "detected unstructured data but not 1-D lat/lon "
                f"(got lat dims {ctt.lat.dims}, lon dims {ctt.lon.dims}), "
                "not pre-computing the triangulation",
                stacklevel=2,
            )

    f = functools.partial(
        _identify_one,
        ctt_threshold=ctt_threshold,
        ctt_core_threshold=ctt_core_threshold,
        size_filter=size_filter,
        size_threshold=size_threshold,
        convex_hull=convex_hull,
        unstructured=unstructured,
        triangulation=triangulation,
    )

    if (not unstructured and len(dims) == 2) or (unstructured and len(dims) == 1):
        if parallel:
            logger.debug(f"assuming one time, ignoring parallel={parallel}")

        ces = [f(ctt)]

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

            logger.info(f"identifying in parallel over {len(itimes)} time steps")
            ces = joblib.Parallel(n_jobs=-2, verbose=10)(
                joblib.delayed(f)(ctt.isel(time=i)) for i in itimes
            )

        else:
            ces = [f(ctt.isel(time=i)) for i in itimes]

    else:
        raise ValueError(
            f"Got unexpected `ctt` dims: {dims}. "
            "They should be 2-D (lat/lon) + optional 'time' dim for structured grid data, "
            "or 1-D (cell) + optional 'time' dim for unstructured grid data."
        )

    inds_empty = [i for i, ce in enumerate(ces) if ce.empty]
    if len(inds_empty) == len(ces):
        warnings.warn("No CEs identified", stacklevel=2)
    elif inds_empty:
        warnings.warn(f"No CEs identified for time steps: {inds_empty}", stacklevel=2)
    logger.info(f"identified CEs in {len(ces) - len(inds_empty)}/{len(ces)} time steps")

    return ces


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

    Raises
    ------
    ValueError
        If the input data is all null or the input frame of shapes is empty.

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

    if contours.empty:
        raise ValueError("Input frame `contours` is empty")

    if isinstance(agg, str):
        agg = (agg,)

    args = (data, contours)
    kwargs = dict(varnames=varnames, agg=agg)

    if method in {"sjoin", "geopandas", "gpd"}:
        new_data = _data_in_contours_sjoin(*args, **kwargs)
    elif method in {"regionmask"}:
        new_data = _data_in_contours_regionmask(*args, **kwargs)
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

    logger = get_worker_logger()

    assert len(contours_sets) == len(times) and len(times) > 1
    times = pd.DatetimeIndex(times)
    itimes = list(range(times.size))

    if durations is not None:
        assert len(durations) == len(times)
        dt = pd.TimedeltaIndex(durations)
    else:
        # Estimate dt values
        dt = times[1:] - times[:-1]
        assert (dt.astype(np.int64) > 0).all()
        if not dt.unique().size == 1:
            warnings.warn("unequal time spacing detected", stacklevel=2)
        dt = dt.insert(-1, dt[-1])

    # IDEA: even at initial time, could put CEs together in groups based on edge-to-edge distance

    if look in {"f", "forward"}:
        warnings.warn("forward `look` considered experimental", stacklevel=2)

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

    # Ensure MCS ID is int (in case of empty times)
    cs["mcs_id"] = cs["mcs_id"].astype(int)

    return cs.reset_index(drop=True)  # drop nested time, CE ind index


class Ellipse(NamedTuple):
    """Ellipse fit parameters, e.g. returned by :func:`fit_ellipse`."""

    center: tuple[float, float]
    """Center ``(x, y)`` coordinates."""

    width: float
    """Diameter in the x-direction before rotation."""

    height: float
    """Diameter in the y-direction before rotation."""

    angle: float
    """Rotation angle (degrees) from x-axis to semi-major axis.
    Positive counter-clockwise."""

    @property
    def a(self) -> float:
        """Semi-major axis length.
        ``width/2`` or ``height/2``, whichever is larger.
        """
        a2, _ = sorted([self.width, self.height], reverse=True)
        return a2 / 2

    @property
    def b(self) -> float:
        """Semi-minor axis length.
        ``width/2`` or ``height/2``, whichever is smaller.
        """
        _, b2 = sorted([self.width, self.height], reverse=True)
        return b2 / 2

    @property
    def c(self) -> float:
        r"""The linear eccentricity (distance from center to focus).

        .. math::
           c = \sqrt{a^2 - b^2}
        """
        return np.sqrt(self.a**2 - self.b**2)

    @property
    def eccentricity(self) -> float:
        r"""The (first) eccentricity.

        .. math::
           e = \sqrt{1 - \frac{b^2}{a^2}}
        """
        return np.sqrt(1 - self.b**2 / self.a**2)

    @property
    def e(self) -> float:
        """Alias for :attr:`eccentricity`."""
        return self.eccentricity

    def to_blob(self):
        """Convert to :class:`~tams.idealized.Blob` object,
        which provides further conversion methods.
        """
        from .idealized import Blob

        return Blob(
            c=self.center,
            a=self.a,
            b=self.b,
            angle=self.angle,
        )


def fit_ellipse(p: shapely.Polygon) -> Ellipse:
    """Fit ellipse to the exterior coordinates of the polygon.

    Raises
    ------
    ValueError
        If ``EllipseModel`` fitting reports failure.

    Notes
    -----
    Using scikit-image ``EllipseModel``
    https://scikit-image.org/docs/dev/api/skimage.measure.html#skimage.measure.EllipseModel

    .. versionadded:: 0.2.0
       In v0.1, the fitted ellipse parameters were used to compute the eccentricity
       but not exposed.
    """
    from skimage.measure import EllipseModel

    xy = np.asarray(p.exterior.coords)
    assert xy.shape[1] == 2

    with warnings.catch_warnings():
        # Current usage is deprecated as of v0.26 (2025-12)
        warnings.filterwarnings(
            "ignore",
            category=FutureWarning,
            message=".* deprecated",
        )

        m = EllipseModel()
        success = m.estimate(xy)

        if not success or all(p is None for p in m.params):
            # pre v0.26, success will be False on early exit,
            # but in v0.26, if estimate exits early for one of the defined warnings,
            # success True is returned, but params is not properly defined
            # success is still False for linalg issues
            raise ValueError("unable to fit ellipse to polygon exterior")

        xc, yc, xhw, yhw, theta = m.params
        # xc, yc, a, b, theta; from the docs
        # a with x, b with y (after subtracting the rotation)
        # they are half-widths, not necessarily the a and b semi-axes
        # theta is in radians

    assert isinstance(xc, float)
    assert isinstance(yc, float)

    return Ellipse(
        center=(xc, yc),
        width=xhw * 2,
        height=yhw * 2,
        angle=np.rad2deg(theta),
    )


def eccentricity(p: shapely.Polygon) -> float:
    """Compute the (first) eccentricity of the least-squares best-fit ellipse
    to the coordinates of the polygon's exterior.

    .. versionchanged:: 0.2.0
       Renamed from :func:`calc_ellipse_eccen`.

    See Also
    --------
    fit_ellipse
    """
    try:
        res = fit_ellipse(p)
    except Exception as e:
        warnings.warn(f"ellipse fitting failed for {p}: {e}", stacklevel=2)
        return np.nan
    else:
        return res.eccentricity


def calc_ellipse_eccen(p: shapely.Polygon) -> float:
    """Calculate the (first) eccentricity of the least-squares best-fit ellipse
    to the coordinates of the polygon's exterior.

    .. deprecated:: 0.2.0
       Renamed to :func:`eccentricity`.
    """
    warnings.warn(
        "`calc_ellipse_eccen` has been renamed to `eccentricity` "
        "and will be removed in a future version",
        FutureWarning,
        stacklevel=2,
    )

    return eccentricity(p)


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
    area = time_groups[["area_km2", "area_core_km2"]].sum()

    # Get duration (time resolution of our CE data)
    dt = time_groups["dtime"].apply(_the_unique)
    dur_tot = dt.sum()

    # Compute area-duration criteria
    dur_219_25k = dt[area.area_core_km2 >= 25_000].sum()
    dur_235_50k = dt[area.area_km2 >= 50_000].sum()
    six_hours = pd.Timedelta(hours=6)

    if dur_219_25k >= six_hours:  # organized
        # Compute ellipse eccentricity
        eps = time_groups[["geometry"]].apply(
            lambda g: eccentricity(g.dissolve().geometry.convex_hull.iloc[0])
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
    """Classify the CE groups into MCS classes
    (categorical column ``'mcs_class'`` in the result).

    Raises
    ------
    ValueError
        If the input frame is missing required columns.
    """
    if cs.empty:
        warnings.warn("empty input frame supplied to `classify`", stacklevel=2)
        return cs.assign(mcs_class=None)

    cols = set(cs.columns)
    cols_needed = {"mcs_id", "geometry", "time", "dtime", "area_km2", "area_core_km2"}
    missing = cols_needed - cols
    if missing:
        raise ValueError(
            f"missing these columns needed by the classify algorithm: {sorted(missing)}"
        )

    classes = cs.groupby("mcs_id")[list(cols_needed)].apply(_classify_one)

    return cs.assign(mcs_class=cs.mcs_id.map(classes).astype("category"))


def run(
    ds: xarray.DataArray,
    *,
    ctt_threshold: float = 235,
    ctt_core_threshold: float = 219,
    u_projection: float = 0,
    parallel: bool = True,
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
    ctt_threshold
        Used to identify the edges of cloud elements.
    ctt_core_threshold
        Used to identify deep convective cloud regions within larger cloud areas.
        This is used to determine whether or not a system is eligible for being classified
        as an organized system.
        It helps target raining clouds.
    u_projection
        *x*\-direction projection velocity to apply before computing overlaps.
    parallel
        Whether to apply parallelization (where possible).

    See Also
    --------
    :func:`tams.identify`
    :func:`tams.track`
    :func:`tams.classify`

    :doc:`/examples/tams-run`
    """
    import itertools

    import geopandas as gpd
    from shapely import MultiPolygon
    from shapely.errors import ShapelyDeprecationWarning

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
    ces = identify(
        ds.ctt,
        ctt_threshold=ctt_threshold,
        ctt_core_threshold=ctt_core_threshold,
        parallel=parallel,
    )

    #
    # 2. Track
    #

    msg("Starting `track`")
    times = ds.time.values
    dt = pd.Timedelta(times[1] - times[0])  # TODO: equal spacing check here?
    ce = track(ces, times, u_projection=u_projection)

    #
    # 3. Classify
    #

    msg("Starting `classify`")
    ce = classify(ce)

    #
    # 4. Stats (including precip)
    #

    if ce.empty:
        raise RuntimeError("no MCSs, unable to compute stats")

    msg("Starting statistics calculations")

    # Cleanup
    ce = ce.drop(columns=["itime", "dtime"]).convert_dtypes()
    ce.core = ce.core.set_crs("EPSG:4326")  # TODO: ensure set in `identify`

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
            d["geometry"] = gpd.GeoSeries(
                time_group[["geometry"]].apply(lambda g: MultiPolygon(g.geometry.values))
            )
            d["core"] = gpd.GeoSeries(
                time_group[["core"]].apply(
                    lambda g: MultiPolygon(
                        itertools.chain.from_iterable(
                            mp.geoms if isinstance(mp, MultiPolygon) else [mp]
                            for mp in g.core.values
                        )
                    )
                )
            )

        d["nce"] = time_group.size()  # codespell:ignore nce
        d["area_km2"] = time_group.area_km2.sum()
        d["area_core_km2"] = time_group.area_core_km2.sum()
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
        .set_geometry("geometry", crs="EPSG:4326")
        .convert_dtypes()
    )
    mcs.core = mcs.core.set_crs("EPSG:4326")
    mcs.mcs_class = mcs.mcs_class.astype("category")

    # Add CTT and PR data stats (time-resolved)
    msg("Starting gridded data aggregation")

    def _agg_one(ds_t, g):
        df1 = data_in_contours(ds_t.pr, g, merge=True)
        df2 = data_in_contours(
            ds_t.pr,
            g.set_geometry("core").drop(columns=["geometry"]),
            merge=False,
        ).add_suffix("_core")
        df3 = data_in_contours(
            ds_t.ctt,
            g.set_geometry("core").drop(columns=["geometry"]),
            merge=False,
        ).add_suffix("_core")
        df = (
            df1.join(df2)
            .join(df3)
            .drop(
                columns=[
                    "count_pr_core",
                ]
            )
            .rename(columns={"count_pr": "npixel", "count_ctt_core": "npixel_core"})
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
    # TODO: these should be duration-weighted, in case dt is not constant (or missing times filled)
    msg("Computing stats for MCS summary dataset")
    vns = [
        "mean_pr",
        "mean_pr_core",
        "mean_ctt_core",
        "std_ctt_core",
        "area_km2",
        "area_core_km2",
        "nce",  # codespell:ignore nce
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

    mcs_summary_points = gpd.GeoDataFrame(
        mcs.groupby("mcs_id")[["geometry", "time"]].apply(f).astype("geometry")
    )
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
