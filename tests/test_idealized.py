import numpy as np

from tams.idealized import Blob


def test_merge_area_con():
    d = 0.5
    b1 = Blob(center=(-d, 0), width=1)
    b2 = Blob(center=(d, 0), width=1)
    merged = b1.merge(b2)

    wh_sum = b1.width * b1.height + b2.width * b2.height
    wh_merged = merged.width * merged.height

    assert np.isclose(wh_sum, wh_merged)
