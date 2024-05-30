import numpy as np
import porespy as ps
from edt import edt as edist
from typing import Tuple, Literal

__all__ = [
    "slicewise_edt",
    "edt",
    "sdt",
    "mis",
    "slicewise_mis",
    "chords",
    "time_of_flight",
    "constriction_factor",
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
    # ? why do we pad this?
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
        # chords = ps.filters.apply_chords_3D(image)
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


