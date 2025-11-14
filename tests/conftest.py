import pytest

import tams


@pytest.fixture
def msg_tb0():
    """Tb from first time of the ``msg-tb`` example dataset."""
    return tams.data.open_example("msg-tb")["tb"].isel(time=0).load()


@pytest.fixture
def mpas():
    """MPAS regridded example dataset."""
    return tams.data.open_example("mpas-regridded")
