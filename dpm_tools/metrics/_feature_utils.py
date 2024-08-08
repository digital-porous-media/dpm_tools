import numpy as np
import cc3d
from scipy.ndimage import binary_erosion, binary_dilation
from skimage.morphology import ball
from typing import Tuple, Any

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


def _sigmoid(x: Any) -> Any:
    """
    Utility function to get the sigmoid function

    Parameters:
        x: Input value

    Returns:
        Any: Sigmoid of the input value(s)
    """
    return 1 / (1 + np.exp(-x))


def _morph_drain_config(image: np.ndarray, radius: float) -> Tuple[np.ndarray, float]:
    """
    Estimate the multiphase configuration of the pore space using the morphological drainage method following
    Hilper & Miller (2001):

    1. Perform erosion for the pore space with specified radius
    2. Label the eroded pore space, and leave only the pore space that is still
       connected with the non-wetting phase.
    3. Perform the dilation for the labelled pore space with specified radius

    This method was adapted from LBPM (https://github.com/OPM/LBPM/blob/master/workflows/Morphological_Analysis/morphological_analysis_utils.py)

    Parameters:
        image: Binary 3D image with values 0 and 1 representing grain space and pore space, respectively
        radius: radius of the structuring element used for erosion and dilation

    Returns:
        Tuple[np.ndarray, float]: A 3D image with estimated multiphase configuration and the wetting saturation at the
        specified invasion radius.
    """

    seg_image = image > 0.0
    pore_vol = np.count_nonzero(seg_image)

    seg_image_padded = np.pad(seg_image.astype(np.uint8), pad_width=1, mode='constant', constant_values=1)
    strel = ball(radius)

    eroded_image = binary_erosion(seg_image_padded, structure=strel)[1:-1, 1:-1, 1:-1]

    eroded_connected_component_labels, num_features = cc3d.connected_components(eroded_image, connectivity=6,
                                                                                return_N=True)

    label_check = eroded_connected_component_labels[0, eroded_connected_component_labels[0, :] != 0]
    label_check = np.unique(label_check)

    eroded_labels = np.zeros_like(eroded_connected_component_labels, dtype=bool)
    for labels in label_check:
        eroded_labels = np.logical_or(eroded_labels, eroded_connected_component_labels == labels)
    eroded_labels = eroded_labels.astype(np.uint8)
    eroded_labels = np.pad(eroded_labels, pad_width=1, mode='constant', constant_values=0)

    # Step 3: perform dilation on the labelled pore space
    eroded_labels_dilated = binary_dilation(eroded_labels.astype(bool), structure=strel)[1:-1, 1:-1, 1:-1]

    eroded_labels_dilated = eroded_labels_dilated.astype(np.uint8)
    eroded_labels_dilated[np.logical_not(eroded_labels_dilated)] = 2
    eroded_labels_dilated[np.logical_not(seg_image)] = 0

    sw = np.count_nonzero(eroded_labels_dilated == 2) / pore_vol

    return eroded_labels_dilated, sw


def _get_heterogeneity_centers_3d(image_shape: Tuple[int, ...], radius: int,
                                  n_samples_per_radius: int, grid: bool = False) -> np.ndarray:
    """
    A utility function to initialize the grid points centers that are sampled for computing the porosity variance

    Parameters:
        image: 3D image
        radius: radius of moving window
        n_samples_per_radius: Number of samples per radius. This is the number of grid points sampled.
        grid: If True, returns a uniform grid of centers. If False, returns a random set of centers

    Returns:
        np.ndarray: Array of centers to sample.

    """
    # -------adjust window's radius with image size------
    ss = np.array(image_shape)
    cnd = radius >= ss / 2

    if sum(cnd) == 0:
        mn = np.array([radius, radius, radius])
        mx = ss - radius
    else:
        mn = (cnd * ss / 2) + np.invert(cnd) * radius
        mx = (cnd * mn + cnd) + (np.invert(cnd) * (ss - radius))

    rw_mn, col_mn, z_mn = mn.astype(int)
    rw_mx, col_mx, z_mx = mx.astype(int)
    # ----------------------------------------------------
    if grid:
        centers = _get_heterogeneity_grid_points(image_shape, n_samples_per_radius)

    else:
        # ------random centroids----------------------
        rndx = np.random.randint(rw_mn, rw_mx, n_samples_per_radius)
        rndy = np.random.randint(col_mn, col_mx, n_samples_per_radius)
        rndz = np.random.randint(z_mn, z_mx, n_samples_per_radius)
        centers = np.array([rndx, rndy, rndz]).T
    return centers


def _get_heterogeneity_grid_points(image_shape: Tuple[int, ...], n_samples: int = 1000) -> np.ndarray:
    """
    Gets indices of "no_points" voxels distributed in a grid within array

    Parameters:
        image: 3D image
        n_samples: Number of sample points to compute porosity variance
    return:
        np.ndarray: Uniform grid centers
    """

    x, y, z = image_shape
    size = x * y * z

    n_samples = min(n_samples, size)

    f = 1 - (n_samples / size)
    s = np.ceil(np.array(image_shape) * f).astype('int')

    nx, ny, nz = s
    xs = np.linspace(0, x, nx, dtype='int', endpoint=True)
    ys = np.linspace(0, y, ny, dtype='int', endpoint=True)
    zs = np.linspace(0, z, nz, dtype='int', endpoint=True)

    rndx, rndy, rndz = np.meshgrid(xs, ys, zs)
    rndx = rndx.flatten()
    rndy = rndy.flatten()
    rndz = rndz.flatten()
    centers = np.array([rndx, rndy, rndz]).T[:n_samples]

    return centers


def create_kernel(kernel_size, arrlib):
    """Create a kernel based on the size of the support structure."""
    kernel = arrlib.ones(kernel_size, dtype=np.float64)
    return kernel

def pad_to_size(array, target_shape, pad_mode='symmetric'):
    """
    Pad an array to the given target shape.

    Parameter:
        array: Numpy array to pad
        target_shape: target shape of the padded array
        pad_mode: mode for padding (default: symmetric)

    Return:
        np.ndarray: Padded array
    """
    assert array.ndim == len(target_shape), "Only 2D and 3D arrays are supported"


    min_idx = [(target_shape[i] - array.shape[i])//2 for i in range(array.ndim)]
    max_idx = [target_shape[i] - min_idx[i] - array.shape[i] for i in range(array.ndim)]
    padding_width = tuple([(min_idx[i], max_idx[i]) for i in range(len(min_idx))])

    return np.pad(array, pad_width=padding_width, mode=pad_mode)

def _centered(arr, newshape, arrlib):
    # Return the center newshape portion of the array.
    newshape = arrlib.asarray(newshape)
    currshape = arrlib.array(arr.shape)
    startind = (currshape - newshape) // 2
    endind = startind + newshape
    myslice = [slice(startind[k], endind[k]) for k in range(len(endind))]
    return arr[tuple(myslice)]

