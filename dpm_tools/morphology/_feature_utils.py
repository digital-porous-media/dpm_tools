import numpy as np


def _get_trend(data, inlet_value: float = 2, outlet_value: float = 1, grid_shift: bool = True) -> np.ndarray:
    """
    Get a linear trend through the domain
    Assumes grain voxels are 0
    """
    linear = np.zeros_like(data.image, dtype=np.float32)

    if grid_shift:
        inlet_value = 2 - 1/data.nz
        outlet_value = 1 - 1/data.nz

    tmp = np.linspace(inlet_value, outlet_value, data.nz)


    for tmp_slice, i in enumerate(tmp):
        linear[:, :, tmp_slice] = np.ones((data.nx, data.ny)) * i

    mask = (data.img != 0)

    linear = linear * mask

    return linear

