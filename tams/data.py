"""
Loaders for various data sets.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import xarray as xr

if TYPE_CHECKING:
    from typing import Sequence

    import xarray


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

HERE = Path(__file__).parent


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
        Brightness temperature (same type as `r`).
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


def download_examples(*, clobber: bool = False) -> None:
    """Download the example datasets.

    * Satellite data (EUMETSAT MSG SEVIRI 10.8 μm IR radiance):
      https://drive.google.com/file/d/1nDWGLPzpe_nld_qbsyQcEYJ-KmMKRSqD/view?usp=sharing
    * MPAS regridded data (brightness temperature and precipitation):
      https://drive.google.com/file/d/1iQEAkFp397ZYGfgBJLMZYiE9aGPqx3o-/view?usp=sharing
    * MPAS native output (unstructured grid) data (brightness temperature and precipitation):
      https://drive.google.com/file/d/1Bb9rjyhfSgJyJTuLnwCun3XnkWUOJ248/view?usp=sharing

    .. note::
       `gdown <https://github.com/wkentaro/gdown>`__,
       used for downloading the files,
       is not currently a required dependency of the TAMS Python package,
       although it is included in the conda-forge recipe.
       gdown is available on conda-forge and PyPI as ``gdown``.
       Files < 100 MB can be easily downloaded from Google Drive with ``wget`` or similar,
       but there are some subtleties with larger files.

       Alternatively, you can download the files manually
       using the links above.

    Parameters
    ----------
    clobber
        If set, overwrite existing files. Otherwise, skip downloading.

    See Also
    --------
    tams.data.load_example_ir
    tams.load_example_tb
    tams.load_example_mpas
    tams.load_example_mpas_ug
    """

    files = [
        ("1nDWGLPzpe_nld_qbsyQcEYJ-KmMKRSqD", "Satellite_data.nc"),  # < 100 MB
        ("1iQEAkFp397ZYGfgBJLMZYiE9aGPqx3o-", "MPAS_data.nc"),  # < 100 MB
        ("1Bb9rjyhfSgJyJTuLnwCun3XnkWUOJ248", "MPAS_unstructured_data.nc"),  # > 100 MB
    ]

    try:
        import gdown
    except ImportError as e:
        raise RuntimeError(
            "gdown is required in order to download the example data files. "
            "It is available on conda-forge and PyPI as 'gdown'."
        ) from e

    def download(id_: str, to: str):
        gdown.download(id=id_, output=to, quiet=False)

    for id_, fn in files:
        fp = HERE / fn
        if not clobber and fp.is_file():
            print(f"Skipping {fn} because it already exists at {fp.as_posix()}.")
            continue
        else:
            download(id_, fp.as_posix())


def load_example_ir() -> xarray.DataArray:
    """Load the example satellite infrared radiance data.

    This comes from the EUMETSAT MSG SEVIRI instrument,
    specifically the 10.8 μm channel (ch9).

    This dataset contains 6 time steps of 2-hourly data (every 2 hours):
    2006-09-01 00--10

    See Also
    --------
    tams.data.download_examples
    """

    ds = xr.open_dataset(HERE / "Satellite_data.nc").rename_dims(
        {"num_rows_vis_ir": "y", "num_columns_vis_ir": "x"}
    )

    ds.lon.attrs.update(long_name="Longitude")
    ds.lat.attrs.update(long_name="Latitude")

    # Times are 2006-Sep-01 00 -- 10, every 2 hours
    ds["time"] = pd.date_range("2006-Sep-01", freq="2h", periods=6)

    return ds.ch9


def load_example_tb() -> xarray.DataArray:
    """Load the example derived satellite brightness temperature data.

    This works by first invoking :func:`tams.data.load_example_ir`
    and then applying :func:`tams.data.tb_from_ir`.

    This dataset contains 6 time steps of 2-hourly data (every 2 hours):
    2006-09-01 00--10

    See Also
    --------
    :func:`tams.data.download_examples`
    :func:`tams.data.load_example_ir`
    :func:`tams.data.tb_from_ir`

    :doc:`/examples/sample-satellite-data`

    :doc:`/examples/tracking-options`
    """

    r = load_example_ir()

    return tb_from_ir(r, ch=9)


def load_example_mpas() -> xarray.Dataset:
    r"""Load the example MPAS dataset.

    This is a spatial and variable subset of native MPAS output,
    Furthermore, it has been regridded to a regular lat/lon grid (0.25°)
    from the original 15-km mesh.

    After regridding, it was spatially subsetted so that
    lat ranges from -5 to 40°N
    and lon from 85 to 170°E.
    This domain relates to the PRECIP field campaign
    (:func:`load_mpas_precip`).

    It has ``tb`` (estimated brightness temperature)
    and ``precip`` (precipitation rate, derived by summing the MPAS accumulated
    grid-scale and convective precip variables ``rainnc`` and ``rainc`` and differentiating).

    ``tb`` was estimated using the (black-body) Stefan--Boltzmann law:

    .. math::
       E = \sigma T^4
       \implies T = (E / \sigma)^{1/4}

    where :math:`E` is the OLR (outgoing longwave radiation, ``olrtoa`` in MPAS output)
    in W m\ :sup:`-2`
    and :math:`\sigma` is the Stefan--Boltzmann constant.

    This dataset contains 127 time steps of hourly data:
    2006-09-08 12 -- 2006-09-13 18.

    See Also
    --------
    :func:`tams.data.download_examples`
    :func:`tams.load_example_mpas_ug`

    :doc:`/examples/tams-run`

    :doc:`/examples/tracking-options`

    :doc:`/examples/sample-mpas-ug-data`
    """

    ds = xr.open_dataset(HERE / "MPAS_data.nc").rename(xtime="time")

    # Mask 0 values of T (e.g. at initial time since OLR is zero then)
    ds["tb"] = ds.tb.where(ds.tb > 0)

    ds.lat.attrs.update(long_name="Latitude", units="degree_north")
    ds.lon.attrs.update(long_name="Longitude", units="degree_east")
    ds.tb.attrs.update(long_name="Brightness temperature", units="K")
    ds.precip.attrs.update(long_name="Precipitation rate", units="mm h-1")

    return ds


def load_example_mpas_ug() -> xarray.Dataset:
    r"""Load the example MPAS unstructured grid dataset.

    This is a spatial and variable subset of native 15-km global mesh MPAS output.

    It has been spatially subsetted so that
    lat ranges from -5 to 20°N
    and lon from 85 to 170°E,
    similar to the example regridded MPAS dataset (:func:`load_example_mpas`)
    except for a smaller lat upper bound.

    Like the regridded MPAS dataset, it has hourly
    ``tb`` (estimated brightness temperature)
    and ``precip`` (precipitation rate)
    for the period
    2006-09-08 12 -- 2006-09-13 18.

    Like the regridded MPAS dataset,
    ``tb`` was estimated using the (black-body) Stefan--Boltzmann law:

    .. math::
       E = \sigma T^4
       \implies T = (E / \sigma)^{1/4}

    where :math:`E` is the OLR (outgoing longwave radiation, ``olrtoa`` in MPAS output)
    in W m\ :sup:`-2`
    and :math:`\sigma` is the Stefan--Boltzmann constant.

    See Also
    --------
    :func:`tams.data.download_examples`
    :func:`tams.load_example_mpas`

    :doc:`/examples/sample-mpas-ug-data`
    """

    ds = xr.open_dataset(HERE / "MPAS_unstructured_data.nc").rename(
        Time="time",
        nCells="cell",
        latcell="lat",
        loncell="lon",
    )

    # Mask zero values of Tb
    ds["tb"] = ds.tb.where(ds.tb > 0)

    # Set time (time variable in there just has elapsed hours as int)
    ds["time"] = pd.date_range("2006-Sep-08 12", freq="1h", periods=ds.sizes["time"])

    # Diff accumulated precip to get mm/h
    ds["precip"] = ds.precip.diff("time", label="lower")

    # Add variables attrs
    ds.lat.attrs.update(long_name="Latitude (cell center)", units="degree_north")
    ds.lon.attrs.update(long_name="Longitude (cell center)", units="degree_east")
    ds.tb.attrs.update(long_name="Brightness temperature", units="K")
    ds.precip.attrs.update(long_name="Precipitation rate", units="mm h-1")

    return ds


def load_mpas_precip(paths: str | Sequence[str], *, parallel: bool = False) -> xarray.Dataset:
    """Derive a TAMS input dataset from post-processed MPAS runs for the PRECIP field campaign.

    Parameters
    ----------
    paths
        Corresponding to the post-processed datasets to load from.

        Pass a string glob or a sequence of string paths.
        (Ensure sorted if doing the latter.)

        .. important::
           Currently it is assumed that each individual file corresponds
           to a single time, which is detected from the file name.
    parallel
        If set, do the initial processing (each file) in parallel.
        Currently uses joblib.
    """

    if isinstance(paths, str):
        from glob import glob

        paths = sorted(glob(paths))

    if len(paths) == 0:
        raise ValueError("no paths")

    def load_one(p):
        import re

        try:
            from scipy.constants import sigma
        except ImportError:
            sigma = 5.67037442e-8

        p = Path(p)

        ds = xr.open_dataset(p)
        ds_ = ds[["olrtoa", "rainc", "rainnc"]]
        ds_ = ds_.rename(Time="time")

        # Detect time from file name and assign
        fn = p.name
        m = re.fullmatch(
            r"mpas_init_20[0-9]{8}_valid_(?P<dt>[0-9]{4}-[0-9]{2}-[0-9]{2}_[0-9]{2})_.+\.nc",
            fn,
        )
        s_dt = m.groupdict()["dt"]
        t = pd.to_datetime(s_dt, format="%Y-%m-%d_%H")
        ds_["time"] = ("time", [t])

        # Compute CTT from TOA OLR
        ds_["tb"] = (ds_.olrtoa / sigma) ** (1 / 4)  # TODO: epsilon?
        ds_.tb.attrs.update(
            long_name="Brightness temperature",
            units="K",
            info="Estimated from 'olrtoa' using the S-B law",
        )

        # Combine precip vars
        ds_["aprecip"] = ds_.rainc + ds_.rainnc
        ds_.aprecip.attrs.update(long_name="Accumulated precip", units="mm")

        # Drop other vars
        ds_ = ds_.drop_vars(["olrtoa", "rainc", "rainnc"])

        return ds_

    # Load combined
    if parallel:
        try:
            import joblib
        except ImportError as e:
            raise RuntimeError("joblib required") from e

        dss = joblib.Parallel(n_jobs=-2, verbose=10)(joblib.delayed(load_one)(p) for p in paths)
    else:
        dss = (load_one(p) for p in paths)

    ds = xr.concat(dss, dim="time")

    # Mask 0 values of T (e.g. at initial time since OLR is zero then)
    ds["tb"] = ds.tb.where(ds.tb > 0)

    # Compute precip by diffing the accumulated precip variable
    # In MPAS output, precip outputs are accumulated *up to* the output timestamp
    # Here, we left-label average rain rate over output time step
    t = pd.to_datetime(ds.time.values)
    dt = t[1:] - t[:-1]
    dt_h = dt.total_seconds() / 3600
    da_dt_h = xr.DataArray(
        dims="time",
        data=np.r_[dt_h, np.nan].astype(np.float32),
        coords={"time": ds.time},
    )
    ds["precip"] = ds.aprecip.diff("time", label="lower") / da_dt_h
    ds.precip.attrs.update(long_name="Precipitation rate", units="mm h-1")
    ds = ds.drop_vars(["aprecip"])

    return ds
