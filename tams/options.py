from copy import deepcopy

OPTIONS = {
    "cache_location": None,
}


class set_options:
    """Globally or temporarily (context manager) set options.

    Parameters
    ----------
    cache_location : str or Path, optional
        ``None`` (default) -> ``pooch.os_cache('tams')``.

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
        OPTIONS.update(dct)

    def __enter__(self):
        return

    def __exit__(self, *args):
        self._update(self.old)


def get_options():
    """Get current options (a copy, edits have no effect)."""
    return deepcopy(OPTIONS)
