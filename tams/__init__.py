"""
TAMS - Tracking Algorithm for Mesoscale Convective Systems
"""

__version__ = "0.1.8"

from . import data
from .core import (
    calc_ellipse_eccen,
    classify,
    contours,
    data_in_contours,
    identify,
    overlap,
    project,
    run,
    track,
)
from .options import get_options, set_options
from .util import plot_tracked
