"""
Create and evolve systems of ellipsoidal blobs for demonstrating and testing TAMS.
"""

from __future__ import annotations

from typing import Self

import numpy as np
import xarray as xr
from shapely import affinity
from shapely.geometry import LinearRing, Point, Polygon


class Blob:

    def __init__(
        self,
        c=(0, 0),
        a: float = 0.5,
        *,
        b: float | None = None,
        theta: float = 0,
        depth: float = 20,
    ) -> None:
        """
        Create a blob with center `c` and semi-axes `a` and `b`.

        Parameters
        ----------
        c : array-like, shape (2,)
            Center of the blob. (x, y) (lon, lat) degrees.
        a : float
            Semi-major axis of the blob.
            When `theta` is 0, this is along the x-axis.
        b : float, optional
            Semi-minor axis of the blob. If not provided, `b` is set to `a` (circle).
            In this case, `a` is the radius of the circle.
        theta : float
            Angle of rotation (degrees).
            When `theta` is 0, `a` is along the x-axis.
        depth : float
            Relative to the environment/background, the depth of the center of the blob.
            In TAMS, 235 K cloud-top temperature is used to define cloud elements,
            while 219 K areas are assumed to represent embedded overshooting tops.
        """
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
        self.theta = theta
        self.depth = depth

        self._tendency = {
            "c": np.zeros(2),
            "a": 0.0,
            "b": 0.0,
            "theta": 0.0,
            "depth": 0.0,
        }

    def __repr__(self) -> str:
        return f"{type(self).__name__}(c={self.c}, a={self.a}, b={self.b}, theta={self.theta})"

    @property
    def center(self) -> Point:
        """Return the defined center as a shapely Point."""
        return Point(self.c)

    @property
    def polygon(self) -> Polygon:
        """Return the ellipse as a shapely Polygon."""
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
        """Return the ellipse perimeter as a shapely LinearRing."""
        return self.polygon.exterior

    def set_tendency(self, **kwargs) -> Self:
        """Set in place the tendency of one or more of the ellipse parameters in per hour units.
        (A typical TAMS time step is 1 or 2 hours.)
        For example, 10 m/s ~ 10 m/s * 3600 s/h / (111000 m/deg) = 0.324 deg/h.
        """
        for k, v in kwargs.items():
            if k not in self._tendency:
                raise ValueError(f"Invalid key: {k!r}")
            self._tendency[k] = v
        return self

    def get_tendency(self, key: str | None = None) -> float | dict[str, float]:
        """Get the current tendency of one or all of the ellipse parameters."""
        if key is not None:
            if key not in self._tendency:
                raise ValueError(f"Invalid key: {key!r}")
            return self._tendency[key]
        else:
            return self._tendency.copy()

    def evolve(self, hours: float, /) -> Self:
        """Evolve the blob in place by the given number of hours."""
        for k, v in self._tendency.items():
            setattr(self, k, getattr(self, k) + v * hours)
        return self

    def dz(self, x: float, y: float, /, *, buffer: float = 1) -> float:
        """Compute the relative depth at point (x, y) in the blob.
        Buffer is relative to the semi-major axis `a`.
        """
        p = Point(x, y)
        poly = self.polygon
        buff = poly.buffer(distance=self.a * buffer)
        if not poly.contains(p):
            if buff.contains(p):
                return self.depth * (1 - (self.a - poly.distance(p)))
            else:
                return 0
        else:
            return -self.depth * (1 - (self.a - poly.distance(p)))


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
    """A field of blobs."""

    def __init__(
        self, blobs: list[Blob], lat=(-10, 10, 42), lon=(-20, 20, 82), z0: float = 270
    ) -> None:
        self.blobs = blobs
        self.lat = _to_arr(lat)
        self.lon = _to_arr(lon)
        self.z0 = z0

    def __repr__(self) -> str:
        return f"{type(self).__name__}(blobs={self.blobs})"

    def evolve(self, hours: float, /) -> Self:
        """Evolve the field in place by the given number of hours."""
        for blob in self.blobs:
            blob.evolve(hours)
        return self

    def to_xarray(self) -> xr.DataArray:
        """Convert the field to an xarray DataArray."""
        import xarray as xr

        nx, ny = len(self.lon), len(self.lat)
        ys, xs = np.meshgrid(self.lon, self.lat)
        dz = np.zeros((ny, nx), dtype=float)
        # TODO: a more efficient way to do this, with GeoPandas?
        for i in range(ny):
            for j in range(nx):
                x, y = xs[i, j], ys[i, j]
                for blob in self.blobs:
                    this_dz = blob.dz(x, y)
                    dz[i, j] += this_dz
                    # TODO: optionally only the max dz here, in overlapping blob case

        z = self.z0 + dz

        ds = xr.Dataset(
            {
                "ctt": (("lat", "lon"), z, {"long_name": "cloud-top temperature", "units": "K"}),
            },
            coords={
                "lat": (("lat"), self.lat),
                "lon": (("lon"), self.lon),
            },
        )

        return ds
