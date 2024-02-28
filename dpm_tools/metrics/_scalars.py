import numpy as np
from quantimpy import minkowski as mk
from typing import Tuple
import edt
from ._feature_utils import _morph_drain_config, _get_heterogeneity_centers_3d
import matplotlib.pyplot as plt



def minkowski_2d(image: np.ndarray, **kwargs) -> Tuple[float, float, float]:
    """
    Compute the 2D Minkowski functionals (area, perimeter, Euler Characteristic)

    Parameters:
        image: The binary image where the phase of interest is 1.
        **kwargs: Additional keyword arguments to be passed to the Quantimpy Minkowski functionals function
    Returns:
        Tuple[float, float, float]: Area, perimeter, radius of curvature
    """

    area, perim, curv = mk.functionals(image.astype(bool), **kwargs)
    perim *= 2 * np.pi
    curv *= np.pi

    return area, perim, curv


def minkowski_3d(image: np.ndarray, **kwargs) -> Tuple[float, float, float, float]:
    """
    Compute the 3D scalar Minkowski functionals (volume, surface area, mean curvature, Euler Characteristic)

    Parameters:
        image: The binary image where the phase of interest is 1.
        **kwargs: Additional keyword arguments to be passed to the Quantimpy Minkowski functionals function

    Returns:
        Tuple[float, float, float, float]: Volume, surface area, mean curvature, Euler characteristic
    """

    vol, sa, curv, ec = mk.functionals(image.astype(bool), **kwargs)
    sa *= 8
    curv *= 2 * np.pi ** 2
    ec *= 4 * np.pi / 3

    return vol, sa, curv, ec


# TODO: Minkowski Tensors

def morph_drain(image: np.ndarray, target_saturation: float = 0.1,
                delta_r: float = 0.05, initial_radius: float = 9999) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute the morphological drainage curve to the target saturation.

    Parameters:
        image: A binary image where the pore phase is labeled 1 and the grain phase is labeled 0.
        target_saturation: The target saturation for the morphological drainage curve. Default = 0.1
        delta_r: Factor by which the invasion radius is decreased. Default = 0.05
        initial_radius: Initial guess for the invasion radius. Default = min(initial_radius, maximum Euclidean distance)
    Returns:
        Tuple[numpy.ndarray, numpy.ndarray, np.ndarray]: The invasion radii and their respective wetting saturation.
        3D image of the estimated fluid configurations at the target saturation.
    """

    # Initialize some parameters
    sw_old = 1.
    sw_new = 1.
    sw_diff_old = 1.
    sw_diff_new = 1.

    img_edt = edt.edt(image)
    r_crit_old = r_crit_new = min(initial_radius, img_edt.max())

    radii = []
    sw = []
    config = image.copy()
    # Compute the morphological drainage curve
    while (sw_new > target_saturation) and (r_crit_new > 0.5):
        sw_diff_old = sw_diff_new
        sw_old = sw_new
        r_crit_old = r_crit_new
        r_crit_new -= delta_r * r_crit_old
        radii.append(r_crit_new)
        config, sw_new = _morph_drain_config(image, r_crit_new)
        sw.append(sw_new)
        sw_diff_new = abs(sw_new - target_saturation)
        print(r_crit_new, sw_new)

    if sw_diff_new < sw_diff_old:
        print(f"Final sw: {sw_new}")
        print(f"Final radius: {r_crit_new}")
    else:
        print(f"Final sw: {sw_old}")
        print(f"Final radius: {r_crit_old}")

    return np.array(radii), np.array(sw), config


def heterogeneity_curve(image: np.ndarray, no_radii: int = 50, n_samples_per_radius: int = 50,
                        min_radius: int = np.inf, max_radius: int = np.inf, grid: bool = False) -> Tuple[np.ndarray, np.ndarray]:

    """
    Compute a curve of the porosity variance over n_samples_per_radius points for no_radii size moving windows.
    This provides a rough estimate for quantifying the heterogeneity of the pore space.

    Parameters:
        image: A binary image where the pore phase is labeled 1 and the grain phase is labeled 0.
        no_radii: The number of radii to compute the curve for. Default = 50.
        n_samples_per_radius: The number of samples to compute porosity variance for each radius. Default = 50.
        min_radius: The minimum radius to compute the curve for. Defaults: Maximum of Euclidean distance transform.
        max_radius: The maximum radius to compute the curve for. Default: Maximum of Euclidean distance transform + 100.
        grid: If True, compute the curve on a regular grid. If False, compute the curve using random locations.
        Default = False

    Returns:
        Tuple[numpy.ndarray, numpy.ndarray]: The radii and porosity variance at each radius.
    """

    min_radius = min(min_radius, edt.edt(image).max())
    max_radius = min(max_radius, min_radius + 100)
    assert max_radius > min_radius, "Max radius must be greater than min radius"

    radii = np.linspace(min_radius, max_radius, no_radii, dtype='int')
    variance = np.empty_like(radii, dtype=np.float64)

    for i, r in enumerate(radii):
        cntrs = _get_heterogeneity_centers_3d(image.shape, r, n_samples_per_radius, grid=grid)

        rr, cc, zz = cntrs[:, 0], cntrs[:, 1], cntrs[:, 2]
        mn = np.array([rr - r, cc - r, zz - r])
        rw_mx, col_mx, z_mx = np.array([rr + r + 1, cc + r + 1, zz + r + 1])
        rw_mn, col_mn, z_mn = (mn > 0) * mn

        porosity = np.empty((n_samples_per_radius,), dtype=np.float64)
        for j in range(n_samples_per_radius):
            porosity[j] = np.count_nonzero(image[rw_mn[j]:rw_mx[j], col_mn[j]:col_mx[j], z_mn[j]:z_mx[j]]) / (
                        2 * r + 1) ** 3
        variance[i] = np.var(porosity)

    return radii, variance



