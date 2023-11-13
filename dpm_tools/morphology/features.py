import numpy as np
import porespy as ps
from scipy.ndimage.morphology import distance_transform_edt as edist
from typing import Tuple
from ..__init__ import timer

@timer
def slicewise_edt(data) -> np.ndarray:
    """
    Returns Euclidean distance transform map of each slice individually
    """
    edt = np.zeros_like(data.image, dtype=np.float32)
    for s in range(data.image.shape[2]):
        edt[:, :, s] = edist(data.image[:, :, s])

    return edt

@timer
def edt(data) -> np.ndarray:
    """
    Returns Euclidean distance transform map of the entire 3D image
    """
    return edist(data.image)

@timer
def slicewise_mis(data, **kwargs) -> np.ndarray:
    """
    A function that calculates the slice-wise maximum inscribed sphere (maximum inscribed disk)
    """
    # TODO why do we pad this?
    input_image = np.pad(array=data.image.copy(), pad_width=((0, 0), (0, 0), (0, 1)), constant_values=1)

    # Calculate slice-wise local thickness from PoreSpy
    thickness = np.zeros_like(input_image)
    for img_slice in range(data.nz + 1):
        thickness[:, :, img_slice] = ps.filters.local_thickness(input_image[:, :, img_slice], sizes=40, mode='hybrid',
                                                                divs=4)

    return thickness

@timer
def chords(data, **kwargs) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    A function that calculates the ellipse area based on the chord lengths in slices orthogonal to direction of flow
    Assumes img = 1 for pore space, 0 for grain space
    Returns length of chords in x, y, and ellipse areas
    """
    ellipse_area = np.zeros_like(data.image, dtype=np.float32)

    for i in range(data.nz):
        # Calculate the chords in x and y for each slice in z
        chords_x = ps.filters.apply_chords(im=data.image[:, :, i], spacing=0, trim_edges=False, axis=0)
        chords_y = ps.filters.apply_chords(im=data.image[:, :, i], spacing=0, trim_edges=False, axis=1)

        # Get chord lengths
        sz_x = ps.filters.region_size(chords_x)
        sz_y = ps.filters.region_size(chords_y)

        # Calculate ellipse area from chords
        ellipse_area[:, :, i] = np.pi/4 * sz_x * sz_y

    return sz_x, sz_y, ellipse_area

@timer
def tof(data, boundary: str = 'l', detrend: bool = True) -> np.ndarray:
    """
    Get time of flight map (solution to Eikonal equation) from specified boundary (inlet or outlet)
    Assumes img = 1 in pore space, 0 for grain space
    Assumes flow is in z direction (orthogonal to axis 2)
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

@timer
def constriction_factor(thickness_map: np.ndarray, power: float = None) -> np.ndarray:
    """
    A function that calculates the slice-wise constriction factor from the input thickness map.
    Constriction factor defined as thickness[x, y, z] / thickness[x, y, z+1]
    Padded with reflected values at outlet
    """
    thickness_map = np.pad(thickness_map.copy(), ((0, 0), (0, 0), (0, 1)), 'reflect')

    if power is not None:
        thickness_map = np.power(thickness_map, power)

    constriction_map = np.divide(thickness_map[:, :, :-1], thickness_map[:, :, 1:])

    # Change constriction to 0 if directly preceding solid matrix (is this the right thing to do?)
    constriction_map[np.isinf(constriction_map)] = 0

    # Change constriction to 0 if directly behind solid matrix
    constriction_map[np.isnan(constriction_map)] = 0

    return constriction_map

if __name__ == '__main__':
    sim = np.fromfile('D:/pore_features/sp_micromodel/sim/elecpot.raw')
    plt.imshow(sim[2])
    plt.show()
    # bin = np.fromfile('D:/pore_features/sp_micromodel/sim/segmented.raw')

