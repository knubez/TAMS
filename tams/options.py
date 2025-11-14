import logging
import os
from copy import deepcopy
from pathlib import Path
from typing import TypedDict


class Options(TypedDict):
    cache_location: str | Path | None
    """Path to the cache directory.
    ``None`` (default) -> ``pooch.os_cache('tams')``
    """

    logger_level: int | str | None
    """Logging level for the "tams" logger.
    ``None`` -> ``logging.NOTSET``.
    """

    logger_handler: str | Path | None
    """Logging handler for the "tams" logger.
    Special string values are ``'stderr'`` and ``'stdout'``;
    others are treated as file paths.
    ``None`` -> no handler (default).
    """


OPTIONS: Options = {
    "cache_location": None,
    "logger_level": logging.WARNING,
    "logger_handler": None,
}


class set_options:
    """Globally or temporarily (context manager) set options.

    Parameters
    ----------
    cache_location : str or Path, optional
        ``None`` (default) -> ``pooch.os_cache('tams')``.
    logger_level : int or str, optional
        ``None`` -> ``logging.NOTSET``.
    logger_handler : {'stderr', 'stdout'} or str or Path, optional
        ``None`` -> no handler (default).

    Examples
    --------
    >>> import tams
    >>> with tams.set_options(cache_location="."):
    ...     ds = tams.data.open_example("msg-tb")
    """

    def __init__(self, **kwargs):
        self.old = {}
        for k in kwargs:
            if k not in OPTIONS:
                raise ValueError(f"Unknown option: {k}")
            self.old[k] = OPTIONS[k]
        self._update(kwargs)

    def _update(self, dct):
        from .util import set_logger_handler, set_logger_level

        for k, v in dct.items():
            OPTIONS[k] = v
            if k == "logger_level":
                if v is None:
                    set_logger_level(logging.NOTSET)
                else:
                    set_logger_level(v)
                os.environ["TAMS_WORKER_LOGGER_LEVEL"] = str(v) if v is not None else ""
            elif k == "logger_handler":
                if v == "stderr":
                    set_logger_handler(stderr=True)
                elif v == "stdout":
                    set_logger_handler(stdout=True)
                else:
                    set_logger_handler(file=v)
                os.environ["TAMS_WORKER_LOGGER_HANDLER"] = str(v) if v is not None else ""
            OPTIONS[k] = v

    def __enter__(self):
        return

    def __exit__(self, *args):
        self._update(self.old)


def get_options():
    """Get current options (a copy, edits have no effect)."""
    return deepcopy(OPTIONS)
