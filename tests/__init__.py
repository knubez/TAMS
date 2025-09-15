from pathlib import Path

import earthaccess
import pytest

GLADE_AVAIL = Path("/glade").is_dir()

try:
    # Raising exceptions on login failure is new in v0.14.0 (2025-02-11)
    # https://github.com/nsidc/earthaccess/releases/tag/v0.14.0
    # https://github.com/nsidc/earthaccess/pull/946/files#diff-58622dff7ea01bdbe5d5e0b10dc8d252da4aecccb4e122d3adeb268e227bf155
    from earthaccess.exceptions import LoginAttemptFailure, LoginStrategyUnavailable
except ImportError:
    earthdata_login_failure_allowed = True
else:
    earthdata_login_failure_allowed = False

if earthdata_login_failure_allowed:
    auth = earthaccess.login()
    earthdata = auth.authenticated
else:
    try:
        auth = earthaccess.login(strategy="netrc")
    except (LoginStrategyUnavailable, LoginAttemptFailure):  # no netrc file or invalid
        earthdata = False
    else:
        earthdata = True
skipif_no_earthdata = pytest.mark.skipif(not earthdata, reason="need Earthdata auth")
