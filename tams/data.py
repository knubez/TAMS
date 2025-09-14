"""
Loaders for various data sets.
"""

from __future__ import annotations

import logging
import warnings
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple

import numpy as np
import pandas as pd
import xarray as xr

if TYPE_CHECKING:
    from typing import Any

    import pooch
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

    Reference: https://resources.eumetrain.org/data/2/204/204.pdf page 13

    https://www.eumetsat.int/msg-services

    The `old SEVIRI page <https://web.archive.org/web/20231112220636/https://www.eumetsat.int/seviri>`__
    provides more detailed information about SEVIRI
    (may take some time to load).

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
        tb.attrs.update(
            units="K",
            long_name="brightness temperature",
            channel=ch,
        )
        tb.name = "tb"

    return tb


class _ExampleFile(NamedTuple):
    key: str
    """Key to identify the example file."""

    file_id: str
    """Google Drive file ID."""

    sha256: str
    """Expected (known) SHA256 hash of the file."""

    fname: str | None = None
    """File name to save as. ``{key}.nc`` will be used if not provided."""


_EXAMPLE_FILES: list[_ExampleFile] = [
    _ExampleFile(
        key="msg-rad-v0.1",
        file_id="1nDWGLPzpe_nld_qbsyQcEYJ-KmMKRSqD",
        sha256="42b0677700b527b677b77ad3450838214a79204188a3c15244af7e88fbfb26db",
    ),
    _ExampleFile(
        key="msg-rad",
        file_id="1nXpbX98Hs39ovcQXEy6eaBPVumJPLXah",
        sha256="14ce08ff06e27e75c19e7a32570ee21302717b6bd073cebacac74807d2f0a7cb",
    ),
    #
    _ExampleFile(
        key="mpas-regridded-v0.1",
        file_id="1iQEAkFp397ZYGfgBJLMZYiE9aGPqx3o-",
        sha256="5d35aae75cf6f8598f922d6326fa8d2e45d6751cc67e0a4b1a5e2719bc1635be",
    ),
    _ExampleFile(
        key="mpas-regridded",
        file_id="1ZPuSXNIM8Vu2AF-L_GVTfx93QfiaWxIv",
        sha256="240d10c671d8d35a717f41a8343573447d1410355cf80e22c468c594c3401186",
    ),
    #
    _ExampleFile(
        key="mpas-native-v0.1",
        file_id="1Bb9rjyhfSgJyJTuLnwCun3XnkWUOJ248",
        sha256="40360896d2043030c48dc809c17e1111124c08519a7b6862670daa73b20f00f2",
    ),
    _ExampleFile(
        key="mpas-native",
        file_id="1ynhJM1z4zQrZxe0VCizybP_fH34KNVHt",
        sha256="9e04f279a5fb1016fec7c24cd5e239f3541d5204b2c13d86b7b39f786ffd50ee",
    ),
    #
    _ExampleFile(
        key="mosa-test-1",
        file_id="1-yETpWWd6pwypT-CmUZ15RcDh5dO29iW",
        sha256="ee0fadaf81105c241d580501b19393bae38a683fe1fb37045a917a6849717238",
    ),
    _ExampleFile(
        key="mosa-test-2",
        file_id="11-8SVOpu39e4HI-l3wIdMHLI_cia8hmx",
        sha256="a9b90b3c0d59de298d1174e87f8bc4f577fca869b9e1ca573c29674acc21e2b4",
    ),
    _ExampleFile(
        key="mosa-test-3",
        file_id="1_IAvkWsE6S95lkZwAV1gN8SooajW2Fec",
        sha256="60b3e27cb3c6b45a5f578a8ed5dde18c9313bf5a22ecf3469d476f7e47b24cc7",
    ),
    _ExampleFile(
        key="mosa-test-4",
        file_id="1KoeCyT4qn_qLmZsheIC88si_4cfjwYb4",
        sha256="2759e77d6a600dbec344701899a6430245ecfde363c7ecaacbc5efada5bda2d2",
    ),
]

_EXAMPLE_FILE_DIRECT_LUT = {f.key: f for f in _EXAMPLE_FILES}

_EXAMPLE_FILE_INDIRECT_LUT = {
    "msg-tb": _EXAMPLE_FILE_DIRECT_LUT["msg-rad"],
}

_EXAMPLE_POSTPROC = {
    "msg-rad": lambda ds: ds["ch9"],
    "msg-tb": lambda ds: ds["ch9"].pipe(tb_from_ir, ch=9),
}


def _gdownload(
    url: str,
    output_file: str,
    pooch_instance: pooch.Pooch | None = None,
    *,
    quiet: bool = True,
) -> None:
    """Download a file from Google Drive using gdown.

    Can be used as a custom downloader for pooch.
    `url` should be just the Google Drive file ID.
    """
    from . import __version__

    try:
        import gdown
    except ImportError as e:
        raise RuntimeError(
            "gdown is required in order to auto download the example data files. "
            "It is available on conda-forge and PyPI as 'gdown'."
        ) from e

    gdown.download(
        id=url,
        output=output_file,
        quiet=quiet,
        user_agent=f"tams {__version__}",
    )


def _get_cache_dir() -> Path:
    from .options import OPTIONS

    cache_location = OPTIONS.get("cache_location")
    if cache_location is None:
        try:
            import pooch
        except ImportError as e:
            raise RuntimeError(
                "pooch is required for caching the example data files. "
                "It is available on conda-forge and PyPI as 'pooch'."
            ) from e
        else:
            p = pooch.os_cache("tams")
    else:
        p = Path(cache_location)

    return p


def clear_cache(*, quiet: bool = False):
    """Clear the cache if it exists, leaving an empty directory.

    Parameters
    ----------
    quiet
        Suppress output messages.
    """
    from shutil import rmtree

    cache_dir = _get_cache_dir()
    s_cache_dir = cache_dir.as_posix()

    if not cache_dir.exists():
        if not quiet:
            print(f"Cache directory {s_cache_dir} does not exist")
        return

    files = sorted(p for p in cache_dir.glob("**/*") if p.is_file())
    if not files and not quiet:
        print(f"No files found in cache directory {s_cache_dir}")

    rmtree(cache_dir)
    cache_dir.mkdir()
    if not quiet:
        print("Removed:")
        for p in files:
            print(f"- {p.relative_to(cache_dir).as_posix()}")


def retrieve_example(key: str, *, progress: bool = False) -> Path:
    """Retrieve an example data file using pooch and gdown.

    Parameters
    ----------
    key
        String identifying which example file to retrieve.
    progress
        Show download progress.

    Examples
    --------
    >>> import tams
    >>> path = tams.data.retrieve_example("msg-tb")

    Notes
    -----
    .. versionadded:: 0.2.0
    """
    try:
        import pooch
    except ImportError as e:
        raise RuntimeError(
            "pooch is required in order to auto download the example data files. "
            "It is available on conda-forge and PyPI as 'pooch'."
        ) from e

    try:
        ef = _EXAMPLE_FILE_DIRECT_LUT[key]
    except KeyError:
        s_keys = ", ".join(repr(f.key) for f in _EXAMPLE_FILES)
        raise ValueError(
            f"unknown example file key {key!r}. Available keys are: {s_keys}."
        ) from None

    pooch_logger = pooch.get_logger()
    pooch_logger_level = pooch_logger.level
    pooch_logger.setLevel(logging.WARNING)

    try:
        p = pooch.retrieve(
            url=ef.file_id,
            known_hash=f"sha256:{ef.sha256}",
            fname=ef.fname or f"{ef.key}.nc",
            path=_get_cache_dir(),
            downloader=partial(_gdownload, quiet=not progress),
        )
    finally:
        pooch_logger.setLevel(pooch_logger_level)

    return Path(p)


def open_example(
    key: str,
    *,
    progress: bool = False,
    **kwargs,
) -> xarray.Dataset | xarray.DataArray:
    """Open an example dataset with xarray.

    Parameters
    ----------
    key
        String identifying the example dataset.
    progress
        Show download progress if applicable.
    **kwargs
        Passed to :func:`xarray.open_dataset`.

    Examples
    --------
    >>> import tams
    >>> ds = tams.data.open_example("msg-tb")

    See Also
    --------
    load_example
        Loads the dataset into memory.

    Notes
    -----
    .. versionadded:: 0.2.0
    """
    lut = {**_EXAMPLE_FILE_DIRECT_LUT, **_EXAMPLE_FILE_INDIRECT_LUT}
    try:
        ef = lut[key]
    except KeyError:
        s_keys = ", ".join(repr(k) for k in lut)
        raise ValueError(
            f"unknown example dataset key {key!r}. Available keys are: {s_keys}."
        ) from None

    p = retrieve_example(ef.key, progress=progress)
    post = _EXAMPLE_POSTPROC.get(key, lambda ds: ds)

    return post(xr.open_dataset(p, **kwargs))


def load_example(
    key: str,
    *,
    progress: bool = False,
    **kwargs,
) -> xarray.Dataset | xarray.DataArray:
    """Load an example dataset into memory with xarray.

    Parameters
    ----------
    key
        String identifying the example dataset.
    progress
        Show download progress if applicable.
    **kwargs
        Passed to :func:`xarray.open_dataset`.

    Examples
    --------
    >>> import tams
    >>> ds = tams.data.load_example("msg-tb")

    See Also
    --------
    open_example
        Just opens (:func:`xarray.open_dataset`) the dataset
        (without triggering the load into memory).

    Notes
    -----
    .. versionadded:: 0.2.0
    """
    with open_example(key, progress=progress, **kwargs) as ds:
        return ds.load()


def _time_input_to_pandas(
    time_or_range: Any | tuple[Any, Any],
) -> tuple[pd.Timestamp | None, pd.Timestamp | None]:
    if isinstance(time_or_range, tuple):
        t0_, t1_ = time_or_range
        t0 = pd.to_datetime(t0_)
        t1 = pd.to_datetime(t1_)
    else:  # Assume single time
        t0 = pd.to_datetime(time_or_range)
        t1 = t0
    if not isinstance(t0, pd.Timestamp) or not isinstance(t1, pd.Timestamp):
        raise TypeError(
            "`time_or_range` must be a single time or a tuple of two times "
            "in a format accepted by pandas.to_datetime"
        )
    return t0, t1


def get_mergir(
    time_or_range: Any | tuple[Any, Any],  # TODO: positional-only
    *,
    version: str = "1",
    parallel: bool = False,
    **kwargs,
) -> xarray.Dataset:
    """Stream GPM MERGIR brightness temperature from NASA Earthdata.

    https://disc.gsfc.nasa.gov/datasets/GPM_MERGIR_1/summary

    This is half-hourly ~ 4-km resolution data in the 60°S--60°N latitude band.
    Each netCDF file contains one hour (two half-hourly time steps).

    .. note::
       The Python packages
       `earthaccess <https://earthaccess.readthedocs.io/en/stable/quick-start/>`__
       and `h5netcdf <https://h5netcdf.org/>`__
       are required.
       Additionally, you must have a
       `NASA Earthdata account <https://www.earthdata.nasa.gov/>`__
       (see https://earthaccess.readthedocs.io/en/stable/howto/authenticate/
       for more info).

    Parameters
    ----------
    time_or_range
        Specific time or time range (inclusive) to request.
    version
        Currently '1' is the only option.
    parallel
        Passed to :func:`xarray.open_mfdataset`, telling it to open files in parallel using Dask.
        This may speed up loading if you are requesting more than a few hours,
        especially if you are using ``dask.distributed``.
    **kwargs
        Passed to :func:`earthaccess.login`.

    See Also
    --------
    :doc:`/examples/get`
    """
    import earthaccess

    t0, t1 = _time_input_to_pandas(time_or_range)

    auth = earthaccess.login(**kwargs)
    if not auth.authenticated:
        raise RuntimeError("NASA Earthdata authentication failed")

    short_name = "GPM_MERGIR"
    results = earthaccess.search_data(
        short_name=short_name,
        version=version,
        cloud_hosted=True,
        temporal=(t0, t1),
        count=-1,
    )

    n = len(results)
    if n == 0:
        raise ValueError(f"no results for period=({t0}, {t1}), version={version!r}")
    elif n >= 1:
        files = earthaccess.open(results)
        if n == 1:
            ds = xr.open_dataset(files[0])
        else:
            ds = xr.open_mfdataset(
                files,
                combine="nested",
                concat_dim="time",
                combine_attrs="drop_conflicts",
                parallel=parallel,
            )

    ds = ds.rename_vars({"Tb": "tb"})
    for vn in ds.variables:
        if vn == "time":
            continue
        ds[vn].attrs["long_name"] = ds[vn].attrs["standard_name"].replace("_", " ")

    # Some times are off by sub-seconds, but we know it should be half-hourly
    ds["time"] = ds.time.dt.round("30min")

    # Select request
    ds = ds.sel(time=slice(t0, t1)).squeeze()

    if "FileHeader" in ds.attrs:
        ds.attrs["FileHeader"] = ds.attrs["FileHeader"].strip().replace("\n", "")
    ds.attrs.update(
        ShortName=short_name,
        Version=version,
    )

    return ds


def get_imerg(
    time_or_range: Any | tuple[Any, Any],  # TODO: positional-only
    *,
    version: str = "07",
    run: str = "final",
    parallel: bool = False,
    **kwargs,
) -> xarray.Dataset:
    """Stream GPM IMERG L3 precipitation from NASA Earthdata.

    https://gpm.nasa.gov/data/directory

    This is half-hourly 0.1° (~ 10-km) resolution data.
    Each HDF5 file contains one time step.

    .. note::
       The Python packages
       `earthaccess <https://earthaccess.readthedocs.io/en/stable/quick-start/>`__
       and `h5netcdf <https://h5netcdf.org/>`__
       are required.
       Additionally, you must have a
       `NASA Earthdata account <https://www.earthdata.nasa.gov/>`__
       (see https://earthaccess.readthedocs.io/en/stable/howto/authenticate/
       for more info).

    Parameters
    ----------
    time_or_range
        Specific time or time range (inclusive) to request.
    version
        For example: '06', '07'.
    run
        'early' and 'late' are available in near-realtime;
        'final' is delayed by a few months.
        See what's available at
        `NASA GES DISC <https://disc.gsfc.nasa.gov/datasets?keywords=gpm%20imerg&page=1&temporalResolution=30%20minutes>`__.
    parallel
        Passed to :func:`xarray.open_mfdataset`, telling it to open files in parallel using Dask.
        This may speed up loading if you are requesting more than a few hours,
        especially if you are using ``dask.distributed``.
    **kwargs
        Passed to :func:`earthaccess.login`.

    See Also
    --------
    :doc:`/examples/get`
    """
    import earthaccess

    t0, t1 = _time_input_to_pandas(time_or_range)

    auth = earthaccess.login(**kwargs)
    if not auth.authenticated:
        raise RuntimeError("NASA Earthdata authentication failed")

    try:
        short_name = (
            "GPM_3IMERGHH"
            + {
                "early": "E",
                "late": "L",
                "final": "",
            }[run.lower()]
        )
    except KeyError as e:
        raise ValueError(f"invalid `run` {run!r}") from e

    results = earthaccess.search_data(
        short_name=short_name,
        version=version,
        cloud_hosted=True,
        temporal=(t0, t1),
        count=-1,
    )

    n = len(results)
    if n == 0:
        raise ValueError(
            f"no {short_name} ({run.lower()}) results for period=({t0}, {t1}), version={version!r}"
        )
    elif n >= 1:
        files = earthaccess.open(results)
        if n == 1:
            ds = xr.open_dataset(files[0], group="Grid")
        else:
            ds = xr.open_mfdataset(
                files,
                group="Grid",
                combine="nested",
                concat_dim="time",
                combine_attrs="drop_conflicts",
                parallel=parallel,
            )

    # Convert to normal datetime
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        # `time_unit` is new in v2025.01.2 (2025-01-31)
        # https://docs.xarray.dev/en/stable/whats-new.html#v2025-01-2-jan-31-2025
        # The future default will be us instead of ns
        try:
            ds["time"] = ds.indexes["time"].to_datetimeindex(time_unit="ns")
        except TypeError:
            ds["time"] = ds.indexes["time"].to_datetimeindex()

    if "precipitationCal" in ds:
        ds = ds.rename_vars({"precipitationCal": "precipitation"})
    ds = (
        ds.drop_dims(["nv", "lonv", "latv"])  # bounds
        .rename_vars(
            {
                "precipitation": "pr",
                "randomError": "pr_err",
                "precipitationQualityIndex": "pr_qi",
            }
        )
        .transpose("time", "lat", "lon")
        .squeeze()
    )
    ds = ds.drop_vars(ds.data_vars.keys() - {"pr", "pr_err", "pr_qi"})

    # Clean up attrs
    long_names = {
        "time": "time",
        "lat": "latitude",
        "lon": "longitude",
        "pr": "precipitation rate",
        "pr_err": "RMSE error estimate for precipitation rate",
        "pr_qi": "quality index for precipitation rate",
    }
    ds["pr_qi"].attrs.update(units="1")
    for vn in ds.variables:
        assert isinstance(vn, str)
        if vn == "time":
            continue
        try:
            desc = " ".join(ds[vn].attrs["LongName"].strip().split())
        except KeyError:
            desc = None
        ds[vn].attrs = {
            "units": ds[vn].attrs["units"],
            "long_name": long_names[vn],
        }
        if desc is not None:
            ds[vn].attrs["description"] = desc
    ds.attrs = {
        "GridHeader": ds.attrs["GridHeader"].strip().replace("\n", ""),
        "ShortName": short_name,
        "Version": version,
    }

    return ds
