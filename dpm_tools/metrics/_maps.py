import numpy as np
import porespy as ps
from edt import edt as edist
from typing import Tuple, Literal
from ._feature_utils import _extract_shape
import scipy.ndimage.morphology as morphology
import scipy.ndimage.measurements as measurements
import cc3d

def slicewise_edt(data) -> np.ndarray:
    """
    Compute the Euclidean distance transform map of each slice individually and stacks them into a single 3D array.

    Parameters:
        data: An image dataclass containing the binary image
    Returns:
        numpy.ndarray: Euclidean distance transform map of each slice
    """
    edt = np.zeros_like(data.image, dtype=np.float32)
    for s in range(data.image.shape[2]):
        edt[:, :, s] = edist(data.image[:, :, s])

    return edt


def edt(data) -> np.ndarray:
    """
    Compute the 3D Euclidean distance transform map of the entire image.

    Parameters:
        data: An image dataclass containing the binary image where the phase of interest is 1.

    Returns:
        numpy.ndarray: 3D Euclidean distance transform map of the entire image
    """
    return edist(data.image)

def sdt(data) -> np.ndarray:
    """
    Signed distance transform where positive values are into the pore space and negative values are into the grain space.

    Parameters:
        data: An image dataclass containing the binary image. Assumes img = 1 for pore space, 0 for grain space.

    Returns:
        numpy.ndarray: Signed distance transform of the entire image
    """
    pores = edist(data.image)
    image_tmp = -1 * data.image.copy() + 1
    grain = -1 * edist(image_tmp)
    signed_distance = grain + pores

    return signed_distance

def mis(data, **kwargs) -> np.ndarray:
    """
    Compute Maximum Inscribed Sphere of the entire image using PoreSpy.

    Parameters:
        data: An image dataclass containing the binary image

    Returns:
        numpy.ndarray: Maximum inscribed sphere of the full 3D image
    """
    input_image = np.pad(array=data.image.copy(), pad_width=((0, 0), (0, 0), (0, 1)), constant_values=1)
    return ps.filters.local_thickness(input_image, **kwargs)


def slicewise_mis(data, **kwargs) -> np.ndarray:
    """
    Compute the slice-wise maximum inscribed sphere (maximum inscribed disk) using PoreSpy.

    Parameters:
        data: An image dataclass containing the binary image

    Returns:
        numpy.ndarray: Maximum inscribed sphere computed on each slice (maximum inscribed disk)
    """
    # ? why do we pad this?
    input_image = np.pad(array=data.image.copy(), pad_width=((0, 0), (0, 0), (0, 1)), constant_values=1)

    # Calculate slice-wise local thickness from PoreSpy
    thickness = np.zeros_like(input_image)
    for img_slice in range(data.nz + 1):
        thickness[:, :, img_slice] = ps.filters.local_thickness(input_image[:, :, img_slice], **kwargs)

    return thickness


def chords(data) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute the ellipse area based on the chord lengths in slices orthogonal to direction of flow
    Assumes img = 1 for pore space, 0 for grain space

    Parameters:
        data: An image dataclass containing the binary image

    Returns:
        Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]: length of chords in x, y, and ellipse areas
    """
    ellipse_area = np.zeros_like(data.image, dtype=np.float32)
    sz_x = np.zeros_like(ellipse_area)
    sz_y = np.zeros_like(ellipse_area)
    for i in range(data.nz):
        # Calculate the chords in x and y for each slice in z
        chords_x = ps.filters.apply_chords(im=data.image[:, :, i], spacing=0, trim_edges=False, axis=0)
        chords_y = ps.filters.apply_chords(im=data.image[:, :, i], spacing=0, trim_edges=False, axis=1)

        # Get chord lengths
        sz_x = ps.filters.region_size(chords_x)
        sz_y = ps.filters.region_size(chords_y)

        # Calculate ellipse area from chords
        ellipse_area[:, :, i] = np.pi / 4 * sz_x * sz_y

    return sz_x, sz_y, ellipse_area


def time_of_flight(data, boundary: Literal['l', 'r'] = 'l', detrend: bool = True) -> np.ndarray:
    """
    Compute time of flight map (solution to Eikonal equation) from specified boundary (inlet or outlet)
    Assumes img = 1 in pore space, 0 for grain space
    Assumes flow is in z direction (orthogonal to axis 2)

    Parameters:
        data: An image dataclass containing the binary image
        boundary: Left ('l') or right ('r') boundaries corresponding to the inlet or outlet
        detrend: If ``detrend`` is True, then subtract the solution assuming no solid matrix is present in the image.
        The solid matrix is still masked.

    Returns:
        numpy.ndarray: Time of flight map from specified boundary
    """
    inlet = np.zeros_like(data.image)
    if boundary[0].lower() == 'l':
        inlet[:, :, 0] = 1.
    elif boundary[0].lower() == 'r':
        inlet[:, :, -1] = 1.
    else:
        raise KeyError("Invalid inlet boundary")

    # Calculate ToF of input image
    tof_map = ps.tools.marching_map(data.image, inlet)

    if detrend:
        tmp = np.ones_like(data.image)
        trend = ps.tools.marching_map(tmp, inlet)
        trend *= data.image  # Mask trended image to obey solid matrix
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


def morphological_drainage_3d(data, R_critical) -> Tuple[np.ndarray, float]:
    from time import perf_counter_ns
    from skimage.morphology import ball
    import porespy as ps

    # The method for this function follows Hilper & Miller AWR(2001)
    # 1. Perform erosion for the pore space with radius of R_critical
    # 2. Label the eroded pore space, and leave only the pore space that is still
    #    connected with the non-wetting phase reservoir
    # 3. Perform the dilation for the labelled pore space with radius of R_critical
    # **************************************************************************
    # Input: seg_image: a well shaped segmented image with size (lz,ly,lx)
    #        seg_image has values as : NW phase -> 1
    #                                   W phase -> 2
    #                               solid phase -> 0
    # **************************************************************************
    # data.image = edist(data.image)
    # if data.image.ndim == 2:
    #
    #     seg_image = data.image > 0.0
    #     pore_vol = 1.0 * seg_image.sum()
    #     radius = R_critical
    #
    #
    #     # Step 1.1: Create structuring element
    #     domain_size = int(np.rint(radius * 2) + 2)
    #     grid = np.indices((domain_size, domain_size))
    #     mk_circle = (grid[0] - domain_size / 2) ** 2 + (grid[1] - domain_size / 2) ** 2 <= radius ** 2
    #     circle = np.zeros((domain_size, domain_size), dtype=np.uint8)
    #     circle[mk_circle] = 1
    #     circle = _extract_shape(circle).astype(bool)
    #
    #     # Step 1.2: Perform erosion on the pore space
    #     # NOTE: the dtype of 'seg_im_ero' is 'bool'
    #     seg_im_ero = morphology.binary_erosion(seg_image, structure=circle, border_value=1)
    #     # NOTE: 'border_value' for erosion should be 'True'
    #
    #     # Step 2: Label the eroded pore space
    #     # NOTE: Assume the NW phase reservoir is at the first layer of the domain
    #     #       i.e. at seg_image[0,:] - adjust it if this does not suit your need
    #     # For erosion, assume that diagonals are not considered
    #     # For erosion, assume that diagonals are not considered
    #     seg_im_ero_label_temp, num_features = measurements.label(seg_im_ero,
    #                                                              structure=morphology.generate_binary_structure(2,
    #                                                                                                             1))
    #     # seg_im_ero_label_temp,num_features = measurements.label(seg_im_ero,structure=morphology.generate_binary_structure(2,2))
    #     # NOTE: Here I assume the inlet is at the first layer of the array's axis=2 (i.e. domain[0,:,:])\
    #     #       You can always change to any other layers as the inlet for this drainage.
    #     label_check = seg_im_ero_label_temp[0, seg_im_ero_label_temp[0, :] != 0]
    #     label_check = np.unique(label_check)

        # NOTE the following lines are only for you to check things
    #     # ******************** For check *******************************#
    #     # It assign the labelled array as: NW -> 1, W -> 2, Solid -> 0
    #     # seg_im_ero_label_show = seg_im_ero_label.copy()
    #     # seg_im_ero_label_show[seg_im_ero_label_show !=1] = 2
    #     # seg_im_ero_label_show[np.logical_not(seg_image_2d)]=0
    #     # ******************** End: for check **************************#
    #
    #     seg_im_ero_label = np.zeros_like(seg_im_ero_label_temp, dtype=bool)
    #     for labels in label_check:
    #         seg_im_ero_label = np.logical_or(seg_im_ero_label, seg_im_ero_label_temp == labels)
    #     seg_im_ero_label = seg_im_ero_label.astype(np.uint8)
    #
    #     # Step 3: perform dilation on the labelled pore space
    #     seg_im_ero_label_dil = morphology.binary_dilation(seg_im_ero_label, structure=circle, border_value=0)
    #     # NOTE: 'border_value' for dilation should be 'False'
    #     # NOTE: the dtype of 'seg_im_ero_label_dil' is 'bool'
    #     seg_im_ero_label_dil = seg_im_ero_label_dil.astype(np.uint8)
    #     seg_im_ero_label_dil[np.logical_not(seg_im_ero_label_dil)] = 2
    #     seg_im_ero_label_dil[np.logical_not(seg_image)] = 0
    #
    #     Sw = (seg_im_ero_label_dil == 2).sum() / pore_vol
    # else:  # 3D porous medium

    seg_image = data.image > 0.0
    pore_vol = 1.0 * seg_image.sum()
    radius = R_critical
    tic = perf_counter_ns()
    # Step 1.1: Create structuring element
    domain_size = int(np.rint(radius * 2) + 2)
    grid = np.indices((domain_size, domain_size, domain_size))
    mk_circle = (grid[0] - domain_size / 2) ** 2 + (grid[1] - domain_size / 2) ** 2 + (
                grid[2] - domain_size / 2) ** 2 <= radius ** 2
    circle = np.zeros((domain_size, domain_size, domain_size), dtype=np.uint8)
    circle[mk_circle] = 1
    circle = _extract_shape(circle).astype(bool)
    toc = perf_counter_ns()
    print(f'Time to create structuring element: {(toc - tic)*1e-9} s')

    # Step 1.2: Perform erosion on the pore space
    # NOTE: the dtype of 'seg_im_ero' is 'bool'
    tic = perf_counter_ns()
    seg_im_ero = morphology.binary_erosion(seg_image, structure=circle, border_value=1)
    # seg_im_ero = ps.filters.fftmorphology(seg_image, strel=ball(radius), mode='erosion')
    # NOTE: 'border_value' for erosion should be 'True'

    toc = perf_counter_ns()
    print(f'Time to perform erosion: {(toc - tic)*1e-9} s')

    # Step 2: Label the eroded pore space
    # NOTE: Assume the NW phase reservoir is at the first layer of the domain
    #       i.e. at seg_image[0,:] - adjust it if this does not suit your need
    # For erosion, assume that diagonals are not considered
    seg_im_ero_label_temp, num_features = measurements.label(seg_im_ero,
                                                             structure=morphology.generate_binary_structure(3,
                                                                                                            1))
    tic = perf_counter_ns()
    # seg_im_ero_label_temp, num_features = cc3d.connected_components(seg_im_ero, connectivity=6, return_N=True)
    # seg_im_ero_label_temp,num_features = measurements.label(seg_im_ero,structure=morphology.generate_binary_structure(3,3))
    # NOTE: Here I assume the inlet is at the first layer of the array's axis=2 (i.e. domain[0,:,:])
    #       You can always change to any other layers as the inlet for this drainage.
    label_check = seg_im_ero_label_temp[0, seg_im_ero_label_temp[0, :] != 0]
    label_check = np.unique(label_check)
    toc = perf_counter_ns()
    print(f'Time to label the eroded pore space: {(toc - tic)*1e-9} s')

    # NOTE the following lines are only for your to check things
    # ******************** For check *******************************#
    # It assign the labelled array as: NW -> 1, W -> 2, Solid -> 0
    # seg_im_ero_label_show = seg_im_ero_label.copy()
    # seg_im_ero_label_show[seg_im_ero_label_show !=1] = 2
    # seg_im_ero_label_show[np.logical_not(seg_image_2d)]=0
    # ******************** End: for check **************************#
    tic = perf_counter_ns()
    seg_im_ero_label = np.zeros_like(seg_im_ero_label_temp, dtype=bool)
    for labels in label_check:
        seg_im_ero_label = np.logical_or(seg_im_ero_label, seg_im_ero_label_temp == labels)
    seg_im_ero_label = seg_im_ero_label.astype(np.uint8)
    toc = perf_counter_ns()
    print(f'Time to check the labels: {(toc - tic)*1e-9} s')
    # Step 3: perform dilation on the labelled pore space
    tic = perf_counter_ns()
    seg_im_ero_label_dil = morphology.binary_dilation(seg_im_ero_label, structure=circle, border_value=0)
    # seg_im_ero_label_dil = ps.filters.fftmorphology(seg_im_ero_label, strel=ball(radius), mode='dilation')
    # NOTE: 'border_value' for dilation should be 'False'
    # NOTE: the dtype of 'seg_im_ero_label_dil' is 'bool'
    seg_im_ero_label_dil = seg_im_ero_label_dil.astype(np.uint8)
    seg_im_ero_label_dil[np.logical_not(seg_im_ero_label_dil)] = 2
    seg_im_ero_label_dil[np.logical_not(seg_image)] = 0
    toc = perf_counter_ns()
    print(f'Time to perform dilation: {(toc - tic)*1e-9} s')

    tic = perf_counter_ns()
    Sw = (seg_im_ero_label_dil == 2).sum() / pore_vol
    toc = perf_counter_ns()
    print(f'Time to calculate the saturation: {(toc - tic)*1e-9} s')
    # end if
    return seg_im_ero_label_dil, Sw

