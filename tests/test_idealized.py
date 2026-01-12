import numpy as np

from tams.idealized import Blob


def test_merge_area_con():
    d = 0.5
    b1 = Blob(c=(-d, 0))
    b2 = Blob(c=(d, 0))
    merged = b1.merge(b2)

    ab_sum = b1.a * b1.b + b2.a * b2.b
    ab_merged = merged.a * merged.b

    assert np.isclose(ab_sum, ab_merged)
