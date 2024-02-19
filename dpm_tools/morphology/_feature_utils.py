import numpy as np

__all__ = ['_set_linear_trend']


def _set_linear_trend(data, inlet_value: float = 2, outlet_value: float = 1, grid_shift: bool = True) -> np.ndarray:
    """
    Set a linear trend through the foreground of a 3D image. Phases labeled 0 will be masked out.
    :param data: Image dataclass
    :param inlet_value: Value to set at the inlet
    :param outlet_value: Value to set at the outlet
    :param grid_shift: If True, shift the inlet and outlet values by 1/nz
    :return: Linear trend through foreground of the image
    :rtype: numpy.ndarray
    """
    linear = np.zeros_like(data.image, dtype=np.float32)

    if grid_shift:
        inlet_value = 2 - 1 / data.nz
        outlet_value = 1 - 1 / data.nz

    tmp = np.linspace(inlet_value, outlet_value, data.nz)

    linear = np.broadcast_to(tmp, data.image.shape)
    # for tmp_slice, i in enumerate(tmp):
    #    linear[:, :, tmp_slice] = np.ones((data.nx, data.ny)) * i

    mask = (data.img != 0)

    linear = linear * mask

    return linear
