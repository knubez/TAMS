"""
WIP: Generate idealized datasets for testing TAMS's
identification and tracking.
"""

from __future__ import annotations

import numpy as np
import xarray as xr


def _peak(
    x,
    y,
    *,
    c: tuple[float, float] = (0, 0),
    h: float = -60,
    A: float = 1,
    zero: bool = True,
):
    """Circular paraboloid.

    Parameters
    ----------
    x, y
        The input grid (2-D position arrays).
    c
        Peak center coordinates.
    h
        Height.
    A
        Scaling factor, applied before shifting up by `h`.
    zero
        Set all values that aren't the same sign as `h` to zero in the returned array.
    """
    xc, yc = c
    z = -1 * np.sign(h) * A * ((x - xc) ** 2 + (y - yc) ** 2) + h
    if zero:
        cond = z >= 0 if h >= 0 else z <= 0
        return np.where(cond, z, 0)
    else:
        return z


def peaks(
    *,
    n: int = 30,
    random_seed: int = 42,
    base: float = 300,
    dx: float = 0,
) -> xr.DataArray:
    """Multiple peaks."""

    # rs = np.random.RandomState(random_seed)

    # Grid
    x_ = np.linspace(-50, 50, 100)
    y_ = np.linspace(-5, 25, 70)
    x, y = np.meshgrid(x_, y_)

    z = np.full_like(x, base)
    # TODO: Poission disk or similar kind of sampling to ensure not touch
    # xcs = rs.uniform(x_[0], x_[-1], n)
    # ycs = rs.uniform(y_[0], y_[-1], n)
    xcs = np.linspace(x_[0], x_[-1], n + 2)[1:-1] + dx
    ycs = np.linspace(y_[0], y_[-1], n + 2)[1:-1]
    for c in zip(xcs, ycs):
        z += _peak(x, y, c=c)

    da = xr.DataArray(
        data=z,
        coords={
            "lat": ("lat", y_),
            "lon": ("lon", x_),
        },
        dims=("lat", "lon"),
    )

    return da


def peaks_ts(*, dx: float = -5, n: int = 3, peaks_kwargs):
    """
    Parameters
    ----------
    dx
        Per time step.
    n
        Number of time steps.
    """
    return xr.concat((peaks(dx=dx * i, **peaks_kwargs) for i in range(n)), dim="time")


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    plt.close("all")

    z = peaks(n=5)

    fig, ax = plt.subplots(figsize=(10, 4))
    z.plot.contourf(shading="auto", cmap="bone_r")  # type: ignore[attr-defined]
    ax.axis("scaled")
    fig.tight_layout()

    peaks_ts(peaks_kwargs=dict(n=3)).plot.pcolormesh(col="time")
