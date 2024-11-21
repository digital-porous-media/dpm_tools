import numpy as np
import porespy as ps
from edt import edt as edist

from ._minkowski_coeff import *
from .feature_utils import pad_to_size, create_kernel, _centered
from ._fft_backends import _get_backend
from ._minkowski_utils import *

from tqdm import tqdm
import skimage
from typing import Tuple, Literal, Any


__all__ = [
    "slicewise_edt",
    "edt",
    "sdt",
    "mis",
    "slicewise_mis",
    "chords",
    "time_of_flight",
    "constriction_factor",
    "minkowski_map"
]

def slicewise_edt(image: np.ndarray) -> np.ndarray:
    """
    Compute the Euclidean distance transform map of each slice individually and stacks them into a single 3D array.

    Parameters:
        image: An np.ndarray containing the binary image with the phase of interest labeled as 1
    Returns:
        numpy.ndarray: Euclidean distance transform map of each slice
    """
    edt = np.zeros_like(image, dtype=np.float32)
    for s in range(image.shape[2]):
        edt[:, :, s] = edist(image[:, :, s])

    return edt


def edt(image) -> np.ndarray:
    """
    Compute the 3D Euclidean distance transform map of the entire image.

    Parameters:
        image: An np.ndarray containing the binary image with the phase of interest labeled as 1

    Returns:
        numpy.ndarray: 3D Euclidean distance transform map of the entire image
    """
    return edist(image)

def sdt(image) -> np.ndarray:
    """
    Signed distance transform where positive values are into the pore space and negative values are into the grain space.

    Parameters:
        image: An np.ndarray containing the binary image. Assumes img = 1 for pore space, 0 for grain space.

    Returns:
        numpy.ndarray: Signed distance transform of the entire image
    """
    pores = edist(image)
    image_tmp = -1 * image.copy() + 1
    grain = -1 * edist(image_tmp)
    signed_distance = grain + pores

    return signed_distance

def mis(image, **kwargs) -> np.ndarray:
    """
    Compute Maximum Inscribed Sphere of the entire image using PoreSpy.

    Parameters:
        image: An np.ndarray containing the binary image with the phase of interest labeled as 1
        **kwargs: Keyword arguments for the ```porespy.filters.local_thickness()``` function

    Returns:
        numpy.ndarray: Maximum inscribed sphere of the full 3D image
    """
    input_image = np.pad(array=image.copy(), pad_width=((0, 0), (0, 0), (0, 1)), constant_values=1)
    return ps.filters.local_thickness(input_image, **kwargs)


def slicewise_mis(image, **kwargs) -> np.ndarray:
    """
    Compute the slice-wise maximum inscribed sphere (maximum inscribed disk) using PoreSpy.

    Parameters:
        image: An np.ndarray containing the binary image with the phase of interest labeled as 1
        **kwargs: Keyword arguments for the ```porespy.filters.local_thickness()``` function

    Returns:
        numpy.ndarray: Maximum inscribed sphere computed on each slice (maximum inscribed disk)
    """
    input_image = np.pad(array=image.copy(), pad_width=((0, 0), (0, 0), (0, 1)), constant_values=1)

    # Calculate slice-wise local thickness from PoreSpy
    thickness = np.zeros_like(input_image)
    for img_slice in range(image.shape[0] + 1):
        thickness[:, :, img_slice] = ps.filters.local_thickness(input_image[:, :, img_slice], **kwargs)

    return thickness


def chords(image) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute the ellipse area based on the chord lengths in slices orthogonal to direction of flow
    Assumes img = 1 for pore space, 0 for grain space

    Parameters:
        image: An np.ndarray containing the binary image with the phase of interest labeled as 1

    Returns:
        Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]: length of chords in x, y, and ellipse areas
    """
    ellipse_area = np.zeros_like(image, dtype=np.float32)
    sz_x = np.zeros_like(ellipse_area)
    sz_y = np.zeros_like(ellipse_area)
    for i in range(image.shape[0]):
        # Calculate the chords in x and y for each slice in z
        chords_x = ps.filters.apply_chords(im=image[i, :, :], spacing=0, trim_edges=False, axis=0)
        chords_y = ps.filters.apply_chords(im=image[i, :, :], spacing=0, trim_edges=False, axis=1)

        # # Get chord lengths
        sz_x[i, :, :] = ps.filters.region_size(chords_x)
        sz_y[i, :, :] = ps.filters.region_size(chords_y)

        # Calculate ellipse area from chords
    ellipse_area = np.pi / 4 * sz_x * sz_y

    return sz_x, sz_y, ellipse_area


def time_of_flight(image, boundary: Literal['l', 'r'] = 'l', detrend: bool = True) -> np.ndarray:
    """
    Compute time of flight map (solution to Eikonal equation) from specified boundary (inlet or outlet)
    Assumes img = 1 in pore space, 0 for grain space
    Assumes flow is in z direction (orthogonal to axis 2)

    Parameters:
        image: An np.ndarray containing the binary image with the phase of interest labeled as 1
        boundary: Left ('l') or right ('r') boundaries corresponding to the inlet or outlet
        detrend: If ``detrend`` is True, then subtract the solution assuming no solid matrix is present in the image.
        The solid matrix is still masked.

    Returns:
        numpy.ndarray: Time of flight map from specified boundary
    """
    inlet = np.zeros_like(image)
    if boundary[0].lower() == 'l':
        inlet[:, :, 0] = 1.
    elif boundary[0].lower() == 'r':
        inlet[:, :, -1] = 1.
    else:
        raise KeyError("Invalid inlet boundary")

    # Calculate ToF of input image
    tof_map = ps.tools.marching_map(image, inlet)

    if detrend:
        tmp = np.ones_like(image)
        trend = ps.tools.marching_map(tmp, inlet)
        trend *= image  # Mask trended image to obey solid matrix
        tof_map -= trend

    return tof_map


def constriction_factor(thickness_map: np.ndarray, power: float = 1.0) -> np.ndarray:
    """
    Compute the slice-wise constriction factor from the input thickness map.
    Constriction factor is defined as thickness[x, y, z] / thickness[x, y, z+1]
    Padded with reflected values at outlet

    Parameters:
        thickness_map: Any 3D thickness map. Examples could be slice-wise EDT, MIS, etc.
        power: Power to raise the thickness map

    Returns:
        numpy.ndarray: Slice-wise constriction factor
    """
    thickness_map = np.pad(thickness_map.copy(), ((0, 0), (0, 0), (0, 1)), 'reflect')

    thickness_map = np.power(thickness_map, power)

    constriction_map = np.divide(thickness_map[:, :, :-1], thickness_map[:, :, 1:])

    # Change constriction to 0 if directly preceding solid matrix (is this the right thing to do?)
    constriction_map[np.isinf(constriction_map)] = 0

    # Change constriction to 0 if directly behind solid matrix
    constriction_map[np.isnan(constriction_map)] = 0

    return constriction_map


def minkowski_map(image: np.ndarray, support_size: list, backend='cpu') -> np.ndarray:
    """
    Compute a map of the 3 (4) Minkowski functionals for a given support size of a 2D (3D) image.

    Method adopted from `Jiang and Arns (2020) <https://journals.aps.org/pre/pdf/10.1103/PhysRevE.101.033302>`_

    Parameters:
        image: 2D or 3D binary image. Foreground voxels should be labeled 1 and background voxels labeled 0.
        support_size: Size of the window to compute local Minkowski functionals.
        backend: Backend for fft computations. Can be 'cpu' or 'cuda'. 'cpu' option uses pyFFTW
    Returns:
         numpy.ndarray: Minkowski maps of size image.shape with local Minkowski functionals

    """

    assert image.ndim == len(support_size), "Image must have same number of dimensions as support_size"
    assert image.ndim == 2 or image.ndim == 3, "Image must be either 2D or 3D"

    if image.dtype != np.uint8:
        image = image.astype(np.uint8)

    if image.ndim == 2:
        return _mink_map_2d(image, support_size, backend)

    elif image.ndim == 3:
        return _mink_map_3d(image, support_size, backend)


def _mink_map_2d(image: np.ndarray, support_size: list, backend: str) -> tuple[Any, Any, Any]:
    fftn, ifftn, to_numpy, get_array, arrlib = _get_backend(backend)
    binary_image_shape = image.shape

    # Pad the binary image such that convolution results in the same size image
    map_shape = tuple([image.shape[i] + support_size[i] - 1 for i in range(len(support_size))])
    # Get next power of 2 size larger than support size
    map_shape = tuple([int(2**np.ceil(np.log2(x))) for x in map_shape])
    image_padded = pad_to_size(image, target_shape=map_shape, pad_mode="constant")

    # Initialize the MF maps
    v2 = arrlib.zeros(binary_image_shape, dtype='float64')
    v1 = arrlib.zeros(binary_image_shape, dtype='float64')
    v0 = arrlib.zeros(binary_image_shape, dtype='float64')

    # Compute the configurations using a convolutional kernel.
    # This will result in an image with the same size as binary image with labels 0-255.
    configs = get_binary_configs_2d(image_padded, image_padded.shape[0], image_padded.shape[1])
    # Define the support structure kernel. This is just an array of ones of size equal to the support structure.
    B = create_kernel(support_size, arrlib)
    B_fft = fftn(B, map_shape, axes=(0, 1))
    B_fft = arrlib.fft.fftshift(B_fft)
    for i in tqdm(range(6)):
        I = get_array(skimage.util.map_array(configs, np.array([i]), np.array([1])))
        I_fft = fftn(I, map_shape, axes=(0, 1))
        I_fft = arrlib.fft.fftshift(I_fft)
        fft_convolution = arrlib.fft.ifftshift(B_fft * I_fft)
        convolution_result = ifftn(fft_convolution, map_shape, axes=(0, 1)).real
        shape_valid = [convolution_result.shape[a] if a not in (0, 1) else binary_image_shape[a]
                       for a in range(convolution_result.ndim)]
        convolution_result = _centered(convolution_result, shape_valid, support_size, arrlib).copy()

        # Minkowski Maps
        v2 += contributions_2d["v2"][i] / 4. * convolution_result
        v1 += contributions_2d["v1"][i] / 8. * np.pi * convolution_result
        # Take the average of 4-connected and 8-connected Euler characteristic
        v0 += (contributions_2d["v0_4"][i] + contributions_2d["v0_8"][i]) * convolution_result / (4. * 2)

    v2, v1, v0 = [to_numpy(v) for v in [v2, v1, v0]]

    return v2, v1, v0


def _mink_map_3d(image: np.ndarray, support_size: list, backend: str) -> tuple[Any, Any, Any, Any]:
    fftn, ifftn, to_numpy, get_array, arrlib = _get_backend(backend)
    binary_image_shape = image.shape

    # Pad the binary image such that convolution results in the same size image
    map_shape = tuple([image.shape[i] + support_size[i] - 1 for i in range(len(support_size))])
    # Get next power of 2 size larger than support size
    map_shape = tuple([int(2**np.ceil(np.log2(x))) for x in map_shape])
    image_padded = pad_to_size(image, target_shape=map_shape, pad_mode="reflect")

    # Initialize the MF maps
    v3 = arrlib.zeros(binary_image_shape, dtype='float64')
    v2 = arrlib.zeros(binary_image_shape, dtype='float64')
    v1 = arrlib.zeros(binary_image_shape, dtype='float64')
    v0 = arrlib.zeros(binary_image_shape, dtype='float64')

    # Compute the configurations using a convolutional kernel.
    # This will result in an image with the same size as binary image with labels 0-255.
    configs = get_binary_configs_3d(image_padded, map_shape[0], map_shape[1], map_shape[2])
    
    # Define the support structure kernel. This is just an array of ones of size equal to the support structure.
    B = create_kernel(support_size, arrlib)
    B_fft = fftn(B, map_shape, axes=(0, 1, 2))
    B_fft = arrlib.fft.fftshift(B_fft)
    for i in tqdm(range(22)):
        I = get_array(skimage.util.map_array(configs, np.array([i]), np.array([1])))
        I_fft = fftn(I, map_shape, axes=(0, 1, 2))
        I_fft = arrlib.fft.fftshift(I_fft)
        fft_convolution = arrlib.fft.ifftshift(B_fft * I_fft)
        convolution_result = ifftn(fft_convolution, map_shape, axes=(0, 1, 2)).real

        shape_valid = [convolution_result.shape[a] if a not in (0, 1, 2) else binary_image_shape[a]
                       for a in range(convolution_result.ndim)]
        convolution_result = _centered(convolution_result, shape_valid, support_size, arrlib).copy()
        v3 += contributions_3d["v3"][i] / 8. * convolution_result
        v2 += contributions_3d["v2"][i] / 24. * 4 * convolution_result
        # Take the average of 4 and 8 connected integral mean curvature
        v1 += (contributions_3d["v1_4"][i] + contributions_3d["v1_8"][i]) * convolution_result / (24. * 2 * np.pi * 2)
        # Take the average of 8-connected and 26-connected Euler characteristic
        v0 += (contributions_3d["v0_6"][i] + contributions_3d["v0_26"][i]) * convolution_result / (8. * 2)

    v3, v2, v1, v0 = [to_numpy(v) for v in [v3, v2, v1, v0]]

    return v3, v2, v1, v0
