"""
Create and evolve systems of ellipsoidal blobs for demonstrating and testing TAMS.

See :doc:`the example notebook <examples/idealized>` for some demonstrations.
"""

from __future__ import annotations

import warnings
from copy import deepcopy
from typing import TYPE_CHECKING

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr
from shapely import affinity
from shapely.geometry import Point

if TYPE_CHECKING:
    from typing import Any

    from matplotlib.patches import Ellipse
    from shapely.geometry import LinearRing, Polygon
    from typing_extensions import Self  # 3.11


class Blob:
    """An elliptical blob that can be used to represent a cloud element
    (or part of one, if blobs are overlapping).

    Parameters
    ----------
    c : array-like of float, shape (2,)
        Center of the blob. (x, y) (lon, lat) degrees.
    a
        Semi-major axis of the blob.
        When `theta` is 0, this is along the x-axis.
    b
        Semi-minor axis of the blob. If not provided, `b` is set to `a` (circle).
        In this case, `a` is the radius of the circle.
    theta
        Angle of rotation (degrees).
        When `theta` is 0, `a` is along the x-axis.
    depth
        Relative to the environment/background, the well depth of the center of the blob.
        Higher depth means a larger negative anomaly.
        In TAMS, 235 K cloud-top temperature is used to define cloud elements,
        while 219 K areas are assumed to represent embedded overshooting tops.
    tendency : dict, optional
        Tendency in any of the blob parameters above (`c`, `a`, `b`, `theta`, `depth`).
        Units: per hour.
        You can also use :meth:`set_tendency` to set the tendency after creating the blob.
        The default tendency is 0 for all parameters.
    """

    def __init__(
        self,
        c=(0, 0),
        a: float = 0.5,
        *,
        b: float | None = None,
        theta: float = 0,
        depth: float = 20,
        tendency: dict[str, Any] | None = None,
    ) -> None:
        self.c = np.asarray(c, dtype=float)
        if not a > 0:
            raise ValueError(f"Invalid semi-major axis: {a}")
        self.a = a
        if b is not None:
            if not 0 < b <= a:
                raise ValueError(f"Invalid semi-minor axis: {b}")
            self.b = b
        else:
            self.b = a
        if self.a == self.b and theta != 0:
            warnings.warn("theta has no effect for circular blobs", stacklevel=2)
        self.theta = theta
        self.depth = depth

        self._tendency = {
            "c": np.zeros(2),
            "a": 0.0,
            "b": 0.0,
            "theta": 0.0,
            "depth": 0.0,
        }
        if tendency is not None:
            self._tendency.update(tendency)

    def __repr__(self) -> str:
        return f"{type(self).__name__}(c={self.c}, a={self.a}, b={self.b}, theta={self.theta})"

    @property
    def center(self) -> Point:
        """The defined center :attr:`c` as a :class:`shapely.Point`."""
        return Point(self.c)

    @property
    def polygon(self) -> Polygon:
        """The ellipse as a :class:`shapely.Polygon`."""
        c = self.center
        r = min(self.a, self.b)
        circle = c.buffer(r)
        if self.a == self.b:
            return circle
        else:
            ellipse = affinity.rotate(
                affinity.scale(circle, xfact=self.a / r, yfact=self.b / r),
                angle=self.theta,
                origin=c,
                use_radians=False,
            )
            return ellipse

    @property
    def ring(self) -> LinearRing:
        """The ellipse perimeter as a :class:`shapely.LinearRing`."""
        return self.polygon.exterior

    def to_geopandas(self, *, crs="EPSG:4326") -> gpd.GeoSeries:
        """Convert the blob to a GeoPandas GeoSeries, using the polygon."""
        return gpd.GeoSeries([self.polygon], crs=crs)

    def to_patch(self, **kwargs) -> Ellipse:
        """Convert the blob to a Matplotlib Ellipse patch.

        Parameters
        ----------
        **kwargs
            Additional keyword arguments to pass to :class:`matplotlib.patches.Ellipse`.
        """
        from matplotlib.patches import Ellipse

        p = Ellipse(
            xy=tuple(self.c),
            width=2 * self.a,
            height=2 * self.b,
            angle=self.theta,
            **kwargs,
        )
        return p

    def set_tendency(self, **kwargs) -> Self:
        """Set in place the tendency of one or more of the ellipse parameters in per hour units.
        (A typical TAMS time step is 1 or 2 hours.)
        For example, 10 m/s ~ 10 m/s * 3600 s/h / (111000 m/deg) = 0.324 deg/h.
        """
        for k, v in kwargs.items():
            if k not in self._tendency:
                raise ValueError(f"Invalid key: {k!r}")
            if k == "c":
                v = np.asarray(v, dtype=float)
            self._tendency[k] = v
        return self

    def get_tendency(self, key: str | None = None, *, copy: bool = False) -> Any | dict[str, Any]:
        """Get the current tendency of one or all (default) of the ellipse parameters."""
        if key is not None:
            if key not in self._tendency:
                raise ValueError(f"Invalid key: {key!r}")
            return self._tendency[key]
        else:
            if copy:
                return deepcopy(self._tendency)
            else:
                return self._tendency

    def evolve(self, hours: float, /) -> Self:
        """Evolve the blob in place by the given number of hours."""
        for k, v in self.get_tendency().items():
            setattr(self, k, getattr(self, k) + v * hours)
        return self

    def well(self, x, y):
        """Parabolic well function in the `a` and `b` directions.

        It reaches `depth` (negative) at the center and 0 at the ellipse edge, then continues.

        Parameters
        ----------
        x, y : array-like
            Two-dimensional array, e.g. from :func:`~numpy.meshgrid`.

        Returns
        -------
        z : array
            The well depth at each :math:`(x, y)` point.
        """
        cx, cy = self.c
        t = np.deg2rad(self.theta)
        dx = x - cx
        dy = y - cy
        r = np.sqrt(dx**2 + dy**2)
        if self.a == self.b:
            # Circular blob
            return -self.depth * (1 - (r / self.a) ** 2)
        else:
            # Elliptical blob
            a = self.a * np.cos(t) + self.b * np.sin(t)
            b = self.a * np.sin(t) + self.b * np.cos(t)
            return -self.depth * (1 - ((dx / a) ** 2 + (dy / b) ** 2))

    def merge(self, other: Blob) -> Blob:
        """Merge with another blob, creating a new blob with area-weighted-average characteristics."""
        if not isinstance(other, Blob):
            raise TypeError(f"Cannot merge {type(other)} with {type(self)}")
        if not self.polygon.intersects(other.polygon):
            raise ValueError("Blobs do not intersect")

        ab_self = self.a * self.b
        ab_other = other.a * other.b
        ab_sum = ab_self + ab_other

        f_self = ab_self / ab_sum
        f_other = ab_other / ab_sum

        # Area-weighted average center
        c = f_self * self.c + f_other * other.c

        # Area-weighted average semi axes
        brel = f_self * self.b / self.a + f_other * other.b / other.a
        a = np.sqrt(ab_sum / brel)
        b = a * brel
        assert np.isclose(ab_sum, a * b), "area conservation"

        # Area-weighted circular average theta
        rad_self = np.deg2rad(self.theta)
        rad_other = np.deg2rad(other.theta)
        theta = np.rad2deg(
            np.arctan2(
                f_self * np.sin(rad_self) + f_other * np.sin(rad_other),
                f_self * np.cos(rad_self) + f_other * np.cos(rad_other),
            )
        )

        # Area-weighted average depth
        depth = f_self * self.depth + f_other * other.depth

        # Area-weighted average tendency
        tendency = {}
        for (k, v_self), (_, v_other) in zip(
            self.get_tendency().items(), other.get_tendency().items()
        ):
            tendency[k] = f_self * v_self + f_other * v_other

        blob = Blob(
            c=c,
            a=a,
            b=b,
            theta=theta,
            depth=depth,
        )
        blob.set_tendency(**tendency)

        return blob

    def split(self, n: int = 2) -> list[Blob]:
        """Split the blob into `n` smaller blobs that line up along the semi-minor axis.
        Or the y-axis if the ellipse is a circle.
        """
        if n < 2:
            raise ValueError(f"n must be at least 2, got {n!r}")

        f = 1 / n
        a, b = f * self.a, f * self.b
        o = self.c
        t = np.deg2rad(self.theta) + np.pi / 2
        blobs = []
        for i in range(n):
            r = 2 * b * (i - (n - 1) / 2)
            c = o + r * np.r_[np.cos(t), np.sin(t)]
            blob = Blob(
                c=c,
                a=a,
                b=b,
                theta=self.theta,
                depth=self.depth,
            )
            blob.set_tendency(**self.get_tendency(copy=True))
            blobs.append(blob)

        return blobs

    def copy(self) -> Blob:
        """Return a copy of the blob."""
        b = Blob(
            c=self.c.copy(),
            a=self.a,
            b=self.b,
            theta=self.theta,
            depth=self.depth,
        )
        b.set_tendency(**self.get_tendency(copy=True))
        return b


def _to_arr(x, *, default_num: int = 100) -> np.ndarray:
    if np.isscalar(x):
        return np.array([x])
    elif isinstance(x, tuple):
        # Assume range spec
        if len(x) == 3:
            return np.linspace(*x)
        elif len(x) == 2:
            return np.linspace(*x, default_num)
        else:
            raise ValueError("tuple must have 2 or 3 elements")
    else:
        # Assume array-like
        return np.asarray(x)


class Field:
    """A field of blobs.

    Parameters
    ----------
    blobs
    lat, lon
        One-dimensional grid coordinate definitions (degrees).
    ctt_background
        Background/environmental/clear-sky infrared brightness temperature.
    """

    def __init__(
        self,
        blobs: Blob | list[Blob] | None = None,
        lat=(-10, 10, 41),
        lon=(-20, 20, 81),
        ctt_background: float = 270,
    ) -> None:
        if blobs is None:
            blobs = []
        elif isinstance(blobs, Blob):
            blobs = [blobs]
        self.blobs = blobs
        self.lat = _to_arr(lat)
        self.lon = _to_arr(lon)
        self.ctt_background = ctt_background

    def __repr__(self) -> str:
        return f"{type(self).__name__}(blobs={self.blobs})"

    def evolve(self, hours: float, /) -> Self:
        """Evolve the field in place by the given number of hours."""
        for blob in self.blobs:
            blob.evolve(hours)
        return self

    def to_xarray(self, *, ctt_threshold: float = 235, additive: bool = False) -> xr.DataArray:
        """Convert the field to an xarray DataArray.

        Parameters
        ----------
        ctt_threshold
            The threshold to be targeted with :func:`tams.identify`.
        additive
            Do the wells (:meth:`Blob.well`) add or min/max?
            The latter (and default) is more consistent with real cloud element merging behavior.
        """
        import xarray as xr

        # The blend region, between blob edge and background
        blend = self.ctt_background - ctt_threshold
        if blend < 0:
            amin, amax = blend, None
            restrict = np.maximum
        elif blend > 0:
            amin, amax = None, blend
            restrict = np.minimum  # type: ignore[assignment]
        else:
            raise ValueError(
                "Blend region must be present. "
                "Set ctt_threshold to a value different from ctt_background."
            )

        nx, ny = len(self.lon), len(self.lat)
        x, y = np.meshgrid(self.lon, self.lat)
        delta = np.zeros((ny, nx), dtype=float)
        for blob in self.blobs:
            if not ((blend > 0 and blob.depth > 0) or (blend < 0 and blob.depth < 0)):
                raise ValueError(
                    "Blob depth and the blend region must have the same sign. "
                    f"Got {blob.depth=} and {blend=}. "
                    "Consider ctt_background and ctt_threshold."
                )
            well = blob.well(x, y)
            this_delta = np.clip(well, amin, amax) - blend
            if additive:
                delta += this_delta
            else:
                delta = restrict(delta, this_delta)

        ctt = self.ctt_background + delta

        da = xr.DataArray(
            data=ctt,
            dims=("lat", "lon"),
            name="ctt",
            attrs={"long_name": "cloud-top temperature", "units": "K"},
            coords={
                "lat": (("lat",), self.lat),
                "lon": (("lon",), self.lon),
            },
        )

        return da

    def to_geopandas(self, *, crs="EPSG:4326") -> gpd.GeoDataFrame:
        """Convert the field to a GeoPandas GeoDataFrame."""
        polygons = [blob.polygon for blob in self.blobs]

        # Create a GeoDataFrame from the polygons
        gdf = gpd.GeoDataFrame(geometry=polygons, crs=crs)

        return gdf

    def copy(self) -> Field:
        """Return a copy of the field."""
        return Field(
            blobs=[blob.copy() for blob in self.blobs],
            lat=self.lat,
            lon=self.lon,
            ctt_background=self.ctt_background,
        )


class Sim:
    """Simulate a field of blobs.

    Parameters
    ----------
    field
        The field (initial state) to start from.
    dt
        Time step (hours).
    """

    def __init__(self, field: Field | None = None, dt: float = 1) -> None:
        if field is None:
            field = Field()
        self.field = field
        if not dt > 0:
            raise ValueError(f"Invalid time step: {dt!r}")
        self.dt = dt
        self._history: list[Field] = []

    def __repr__(self) -> str:
        steps = len(self._history)
        return f"{type(self).__name__}(field={self.field}, steps={steps})"

    def advance(self, steps: int, /) -> Self:
        """Advance the simulation by the given number of time steps."""
        for _ in range(steps):
            self._history.append(self.field.copy())
            self.field.evolve(self.dt)
        return self

    def to_xarray(self, *, start="2006-09-08 12:00", **kwargs) -> xr.DataArray:
        """Convert the history and current field to an xarray DataArray.

        Parameters
        ----------
        start : datetime-like
            The starting time to use when defining the time coordinate.
            Passed to :func:`pandas.date_range`.
        **kwargs
            Additional keyword arguments to pass to :meth:`Field.to_xarray`.
        """
        das = [field.to_xarray(**kwargs) for field in self._history + [self.field]]
        da = xr.concat(das, dim="time")

        freq = pd.Timedelta(self.dt, unit="h")
        da["time"] = pd.date_range(start=start, freq=freq, periods=len(das))

        return da
