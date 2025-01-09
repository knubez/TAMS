"""
TAMS - Tracking Algorithm for Mesoscale Convective Systems
"""

__version__ = "0.1.6"

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
from .data import load_example_mpas, load_example_mpas_ug, load_example_tb, load_mpas_precip
from .util import plot_tracked
