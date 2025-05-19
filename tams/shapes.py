"""
Shapely shape helpers.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Iterable, Union

import numpy as np
from shapely.geometry import LineString, MultiPolygon, Point, Polygon
from shapely.ops import polygonize, unary_union

if TYPE_CHECKING:
    from shapely import Geometry

__all__ = (
    "make_arc",
    "make_circle",
    "make_cone",
    "make_cone2",
    "make_ellipse",
    "make_line",
    "make_rectangle",
    "make_rectangle2",
    "make_square",
    "split",
)

PointLike = Union[Iterable[float], Point]
"""
A 2 (or 3, technically) element sequence of x, y[, z] coords or a :class:`shapely.Point` instance.
"""


def make_line(p1: PointLike, p2: PointLike) -> LineString:
    """Return the line connecting the two points."""

    return LineString([p1, p2])


def make_arc(
    xy: PointLike,
    r: float,
    angle_range: tuple[float, float],
    *,
    n: int | None = None,
) -> LineString:
    """Make an arc line for angle a->b (degrees), with origin `xy` and radius `r`.

    Parameters
    ----------
    n
        Number of points to use.
        Defaults to the angle range (but clipped to 10--360).
    """
    # Based on https://stackoverflow.com/a/30762727
    from math import ceil

    p = Point(xy)

    a, b = angle_range
    while b < a:
        b += 360
    if a == b:
        raise ValueError

    if n is None:
        n = np.clip(ceil(abs(b - a)), 10, 360)

    theta = np.deg2rad(np.linspace(a, b, n))
    x = p.x + r * np.cos(theta)
    y = p.y + r * np.sin(theta)

    return LineString(np.column_stack((x, y)))


def make_rectangle(p1: PointLike, p2: PointLike) -> Polygon:
    """Return the rectangle defined by two points forming a diagonal
    (e.g. the lower-left and upper-right corners)."""

    p1, p2 = Point(p1), Point(p2)
    if p1.x == p2.x or p1.y == p2.y:
        raise ValueError("not a diagonal")

    return Polygon([(p1.x, p1.y), (p1.x, p2.y), (p2.x, p2.y), (p2.x, p1.y)])


def make_rectangle2(xy: tuple[float, float], w: float, h: float) -> Polygon:
    """Return the rectangle centered on `xy` with width `w` and height `h`."""

    x, y = xy
    if not w > 0 and h > 0:
        raise ValueError("w and h must be positive")
    hw, hh = w / 2, h / 2

    return Polygon(
        [
            (x - hw, y - hh),
            (x - hw, y + hh),
            (x + hw, y + hh),
            (x + hw, y - hh),
        ]
    )


def make_square(xy: tuple[float, float], s: float) -> Polygon:
    """Return the square centered on `xy` with side length `s`."""

    if not s > 0:
        raise ValueError("s must be positive")

    return make_rectangle2(xy, s, s)


def make_circle(
    xy: PointLike,
    d: float,
    *,
    half: bool = True,
) -> Polygon:
    """
    Parameters
    ----------
    xy
        Center point.
    d
        Diameter.
        (Or, radius if `half` is true.)
    half
        Whether `d` is diameter (``False``, default)
        or half-diameter (i.e., radius; ``True``).
    """
    from shapely.geometry import Point

    r = d if half else d / 2

    # TODO: increase `quad_segs` for large `r`?
    return Point(xy).buffer(r)


def make_ellipse(
    xy: PointLike,
    w: float,
    h: float,
    *,
    angle: float = 0,
    half: bool = False,
) -> Polygon:
    """
    Parameters
    ----------
    xy
        Center point.
    w, h
        Ellipse width and height.
        (Or, half-width and half-height, the semi-diameters, if `half` is true.)
    angle
        Rotation (counter-clockwise; applied after creation; **degrees**).
        For example ``angle=90`` will cause width and height to be switched in the result.
    half
        Whether `w` and `h` are expressed as
        full diameters (``False``, default)
        or semi/half diameters (``True``).
    """
    # Based on https://gis.stackexchange.com/a/243462
    import shapely.affinity

    if half:
        hw, hh = w, h
    else:
        hw, hh = w / 2, h / 2

    return shapely.affinity.rotate(
        shapely.affinity.scale(
            Point(xy).buffer(1),
            hw,
            hh,
        ),
        angle,
    )


def make_cone(
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    r_range: tuple[float, float],
) -> Polygon:
    """Make a cone-like shape by combining many overlapping circles of different radii.

    The result resembles an ice cream cone.

    Parameters
    ----------
    x_range, y_range, r_range
        Center and radius of the initial and final circles used to construct the cone.
        For example ``(0, -10), (0, 0), (0.2, 3)`` will create a cone
        with the smallest circle centered at (0, 0) and the largest at (-10, 0),
        like an ice cream cone that has fallen to the left.
    """
    # Based on https://gis.stackexchange.com/a/326692
    n = np.ceil(max(np.ptp(x_range), np.ptp(y_range))).astype(int) * 2
    x = np.linspace(*x_range, n)
    y = np.linspace(*y_range, n)
    r = np.linspace(*r_range, n)

    # Coords
    theta = np.linspace(0, 2 * np.pi, 360)
    polygon_x = x[:, None] + r[:, None] * np.sin(theta)
    polygon_y = y[:, None] + r[:, None] * np.cos(theta)

    # Circles
    ps = [Polygon(i) for i in np.dstack((polygon_x, polygon_y))]

    # Convex hulls of subsequent circles
    n = range(len(ps) - 1)
    convex_hulls = [MultiPolygon([ps[i], ps[i + 1]]).convex_hull for i in n]

    # Final polygon
    cone = unary_union(convex_hulls)
    assert isinstance(cone, Polygon), f"expected Polygon, got {type(cone)}"

    return cone


def make_cone2(xy: tuple[float, float], h: float, d: float, *, rcap=None) -> Polygon:
    """Make a cone-like shape by combining two lines and an arc.

    Compared to the circle-cone (:func:`make_cone`):

    * the default cap is less curved
    * the base (opposite the cap) is a point, instead of a (generally small) circle

    Parameters
    ----------
    xy
        The base point (opposite the cap; the top of the isosceles triangle).
    h
        Cone height (height of the isosceles triangle).
    d
        Cone diameter (cap diameter; base length of the isosceles triangle).
    rcap
        Cap radius.
        Default is the arc from the circle centered on `xy` with radius ``d/2``.
        Use infinity (e.g. ``np.inf``) for a straight line
        (i.e., a normal isosceles triangle).
    """

    x, y = xy
    r = d / 2
    hyp = np.hypot(h, r)
    if rcap is None:
        rcap = hyp

    if rcap < r:
        raise ValueError(f"cap circle {rcap} is too small given {h=} and {d=}")

    s1 = make_line(xy, (x + h, y + r))
    s2 = make_line(xy, (x + h, y - r))

    if np.isinf(rcap):
        cap = make_line((x + h, y + r), (x + h, y - r))
    else:
        t = np.arcsin(d / 2 / rcap)  # half angle for the cap circle
        c = rcap * np.cos(t)
        xc = x + h - c
        tdeg = np.rad2deg(t)
        cap = make_arc((xc, y), rcap, (-tdeg, tdeg))

    (poly,) = polygonize([s1, s2, cap])

    return poly


def split(polygon: Polygon, cutter: Geometry) -> list[Polygon]:
    """Split polygon along line(s).

    Parameters
    ----------
    polygon
        The polygon to cut.
    cutter
        `cutter` could be, for example, a :class:`~shapely.LineString`,
        made with :func:`make_line` or :func:`make_arc`.
        If it is a :class:`~shapely.Polygon`, the :attr:`~shapely.Polygon.boundary` will be used.
    """
    # Based on https://kuanbutts.com/2020/07/07/subdivide-polygon-with-linestring/
    # TODO: cut linestring?
    # TODO: account for polygon holes?

    to_cut = polygon

    # Union the exterior lines of the polygon with the dividing linestring
    unioned = to_cut.boundary.union(cutter)

    # Filter out polygons outside of original input polygon
    polys = [poly for poly in polygonize(unioned) if poly.representative_point().within(to_cut)]

    # TODO: return Polygon/MultiPolygon instead?
    return polys
