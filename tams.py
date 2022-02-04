"""
TAMS
"""
import numpy as np

_tb_from_ir_coeffs = {
    4: (2569.094, 0.9959, 3.471),
    5: (1598.566, 0.9963, 2.219),
    6: (1362.142, 0.9991, 0.485),
    7: (1149.083, 0.9996, 0.181),
    8: (1034.345, 0.9999, 0.060),
    9: (930.659, 0.9983, 0.627),
    10: (839.661, 0.9988, 0.397),
    11: (752.381, 0.9981, 0.576),
}


def tb_from_ir(r, ch: int):
    """Compute brightness temperature from IR satellite radiances (`r`)
    in channel `ch`.

    Reference: http://www.eumetrain.org/data/2/204/204.pdf page 13

    Parameters
    ----------
    r : array-like
        Radiance. Units: m2 m-2 sr-1 (cm-1)-1
    ch
        Channel number, in 4--11.
    """
    if ch not in range(4, 12):
        raise ValueError("channel must be in 4--11")

    c1 = 1.19104e-5
    c2 = 1.43877

    vc, a, b = _tb_from_ir_coeffs[ch]

    tb = (c2 * vc / np.log((c1 * vc ** 3) / r + 1) - b) / a

    tb.attrs.update(units="K", long_name="Brightness temperature")

    return tb


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import xarray as xr

    r = xr.open_dataset("Satellite_data.nc").ch9

    tb = tb_from_ir(r, 9)

    tb.isel(time=0).plot()

    plt.show()
