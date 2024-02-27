import numpy as np
from quantimpy import minkowski as mk
from typing import Tuple


def minkowski_2d(data, **kwargs) -> Tuple[float, float, float]:
    """
    Compute the 2D Minkowski functionals (area, perimeter, Euler Characteristic)

    Parameters:
        data: An image dataclass containing the binary image where the phase of interest is 1.
        **kwargs: Additional keyword arguments to be passed to the Quantimpy Minkowski functionals function
    Returns:
        Tuple[float, float, float]: Area, perimeter, radius of curvature
    """

    minkowski = mk.functionals(data.image.astype(bool), **kwargs)
    minkowski[1] *= 2 * np.pi
    minkowski[2] *= np.pi

    return minkowski


def minkowski_3d(data, **kwargs) -> Tuple[float, float, float, float]:
    """
    Compute the 3D scalar Minkowski functionals (volume, surface area, mean curvature, Euler Characteristic)

    Parameters:
        data: An image dataclass containing the binary image where the phase of interest is 1.
        **kwargs: Additional keyword arguments to be passed to the Quantimpy Minkowski functionals function

    Returns:
        Tuple[float, float, float, float]: Volume, surface area, mean curvature, Euler characteristic
    """

    minkowski = mk.functionals(data.image.astype(bool), **kwargs)
    minkowski[1] *= 8
    minkowski[2] *= 2 * np.pi ** 2
    minkowski[3] *= 4 * np.pi / 3

    return minkowski


# TODO: Minkowski Tensors