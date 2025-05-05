"""
Create and evolve systems of ellipsoidal blobs for demonstrating and testing TAMS.
"""

from __future__ import annotations

from typing import Self

import numpy as np
from shapely import affinity
from shapely.geometry import LinearRing, Point, Polygon


class Blob:

    def __init__(self, c, a: float, *, b: float | None = None, theta: float = 0) -> None:
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
        theta : float, optional
            Angle of rotation (degrees).
            When `theta` is 0, `a` is along the x-axis.
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

        self._tendency = {
            "c": np.zeros(2),
            "a": 0.0,
            "b": 0.0,
            "theta": 0.0,
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
                raise ValueError(f"Invalid tendency: {k}")
            self._tendency[k] = v
        return self

    def evolve(self, hours: float, /) -> Self:
        """Evolve the blob in place by the given number of hours."""
        for k, v in self._tendency.items():
            setattr(self, k, getattr(self, k) + v * hours)
        return self
