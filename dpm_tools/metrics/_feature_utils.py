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


def _extract_shape(domain):
    if domain.ndim == 3:
        where_tube = np.where(domain)
        z_start = where_tube[0].min()
        z_end = where_tube[0].max()
        y_start = where_tube[1].min()
        y_end = where_tube[1].max()
        x_start = where_tube[2].min()
        x_end = where_tube[2].max()
        domain_seg = domain[z_start:z_end+1, y_start:y_end+1, x_start:x_end+1].copy()
        # IMPORTANT: if you have "domain_seg = domain[y_start:yEnd+1,x_start:x_end+1]"
        #            then "domain_seg" is only a view of domain, and later on you have
        #            any changes on your "domain_seg", the "domain" will also be changed
        #            correspondingly, which might introduce unexpected conflicts and errors
    else: # domain.ndim == 2
        where_tube = np.where(domain)
        y_start = where_tube[0].min()
        y_end = where_tube[0].max()
        x_start = where_tube[1].min()
        x_end = where_tube[1].max()
        domain_seg = domain[y_start:y_end+1, x_start:x_end+1].copy()
    return domain_seg
