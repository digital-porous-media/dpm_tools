import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from typing import Tuple, List, Dict
import edt
from ._minkowski_coeff import contributions_2d, contributions_3d
from ._feature_utils import _morph_drain_config, _get_heterogeneity_centers_3d
from ._minkowski_utils import get_configs_histogram_2d, get_configs_histogram_3d


def minkowski_functionals(image: np.ndarray) -> Tuple:
    """
    Compute the 2D or 3D Minkowski functionals from a Numpy array.

    Parameters:
        image: The binary image where the phase of interest is 1. Datatype should be 'uint8'
    Returns:
        Tuple, float: For 2D images, returns a tuple of area, perimeter, and Euler characteristic. For 3D images, returns
        a tuple of volume, surface area, integral mean curvature, and Euler characeteristic.
    """
    if image.dtype != np.uint8:
        image = image.astype(np.uint8)
    if image.ndim == 2:
        return _minkowski_2d(image)
    elif image.ndim == 3:
        return _minkowski_3d(image)
    else:
        raise Exception("Image must be 2D or 3D image")


def _minkowski_2d(image: np.ndarray) -> Tuple[float, float, float]:
    """
    Helper function to compute the 2D Minkowski functionals (area, perimeter, Euler Characteristic)

    Parameters:
        image: The binary image where the phase of interest is 1.
    Returns:
        Tuple[float, float, float]: Area, perimeter, radius of curvature
    """
    image = np.pad(image, ((1, 1), (1, 1)), mode='constant', constant_values=0)

    # Get the isotropic configurations (3D)
    nx, ny = image.shape
    configs_hist = get_configs_histogram_2d(image, nx, ny)
    v2 = np.sum(contributions_2d["v2"] / 4. * configs_hist)
    v1 = np.sum(contributions_2d["v1"] / 8. * np.pi * configs_hist)
    v0_8 = np.sum(contributions_2d["v0_8"] / 4. * configs_hist)
    v0_4 = np.sum(contributions_2d["v0_4"] / 4. * configs_hist)
    v0 = (v0_4 + v0_8) / 2

    return v2, v1, v0


def _minkowski_3d(image: np.ndarray) -> Tuple[float, float, float, float]:
    """
    Helper function to compute the 3D scalar Minkowski functionals (volume, surface area, mean curvature, Euler Characteristic)

    Parameters:
        image: The binary image where the phase of interest is 1.
    Returns:
        Tuple[float, float, float, float]: Volume, surface area, mean curvature, Euler characteristic
    """
    image = np.pad(image, ((1, 1), (1, 1), (1, 1)),
                   mode='constant', constant_values=0)

    # Get the isotropic configurations (3D)
    nx, ny, nz = image.shape
    configs_hist = get_configs_histogram_3d(image, nx, ny, nz)
    v3 = np.sum(contributions_3d["v3"] / 8. * configs_hist)
    v2 = np.sum(contributions_3d["v2"] / 24. * 4 * configs_hist)
    v1_4 = np.sum(contributions_3d["v1_4"] / 24. * 2 * np.pi * configs_hist)
    v1_8 = np.sum(contributions_3d["v1_8"] / 24. * 2 * np.pi * configs_hist)
    # Take the average of 4-connected and 8-connected interfaces
    v1 = (v1_4 + v1_8) / 2

    v0_6 = np.sum(contributions_3d["v0_6"] / 8. * configs_hist)
    v0_26 = np.sum(contributions_3d["v0_26"] / 8. * configs_hist)
    # Take the average of 6-connected and 26-connected interfaces
    v0 = (v0_6 + v0_26) / 2

    return v3, v2, v1, v0


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
        cntrs = _get_heterogeneity_centers_3d(
            image.shape, r, n_samples_per_radius, grid=grid)

        rr, cc, zz = cntrs[:, 0], cntrs[:, 1], cntrs[:, 2]
        mn = np.array([rr - r, cc - r, zz - r])
        rw_mx, col_mx, z_mx = np.array([rr + r + 1, cc + r + 1, zz + r + 1])
        rw_mn, col_mn, z_mn = (mn > 0) * mn

        porosity = np.empty((n_samples_per_radius,), dtype=np.float64)
        for j in range(n_samples_per_radius):
            porosity[j] = np.count_nonzero(
                image[rw_mn[j]:rw_mx[j], col_mn[j]:col_mx[j], z_mn[j]:z_mx[j]]) / (2 * r + 1) ** 3
        variance[i] = np.var(porosity)

    return radii, variance

def _hist_stats(image: np.ndarray, nbins: int = 256) -> Dict:
    """
    Calculate histogram statistics of an image.

    Parameters:
        image: A 2D or 3D image.
        nbins: Number of histogram bins. Defaults to 256.

    Returns:
        dict: A dictionary of image metadata and histogram statistics.
        Includes the shape, dtype, min, max, mean, median, variance, skewness, and kurtosis.
    """
    
    stats_dict = {'shape': image.shape, 
                  'dtype': image.dtype,
                  'min': np.amin(image),
                  'max': np.amax(image),
                  'mean': np.mean(image),
                  'median': np.median(image),
                  'variance': np.var(image)}
    
    hist, bin_edges = np.histogram(image.flatten(), bins=256)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    
    stats_dict['skewness'] = stats.skew(bin_centers)
    stats_dict['kurtosis'] = stats.kurtosis(bin_centers, fisher=True)

    return stats_dict

def histogram_statistics(*images: np.ndarray, plot_histogram: bool = False, nbins: int = 256, legend_elem: List = []) -> List[Dict]:
    """
    Get the statistics of multiple images.

    Parameters:
        plot_histograms: Plot the image histograms. Defaults to False.
        bins: Number of histogram bins. Defaults to 256.

    Returns:
        dict: A dictionary of image metadata and histogram statistics.
        Includes the shape, dtype, min, max, mean, median, variance, skewness, and kurtosis.
    """     
    
    img_stats = []
    img_list = []
    
    for image in images:
        stats = _hist_stats(image, nbins)
        
        print('-'*35)
        print('Image statistics:')
        print(f'\tShape: {stats["shape"]}')
        print(f'\tData type: {stats["dtype"]}')
        print(f'\tMin: {stats["min"]}, Max: {stats["max"]}, Mean: {stats["mean"]}\n')
        
        
        img_stats.append(_hist_stats(image, nbins))
        img_list.append(image.flatten())
    
    if plot_histogram:
        assert len(legend_elem) == len(images), 'Number of legend elements must match the number of images'
        fig, ax = plt.subplots()
        ax.hist(img_list, bins=nbins, density=True, histtype='step', fill=False)
        ax.set_xlabel('Image Value', fontsize=16)
        ax.set_ylabel('Probability Density', fontsize=16)
        ax.grid(True) 
        ax.legend(legend_elem)

    return img_stats