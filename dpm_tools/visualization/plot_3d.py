import numpy as np
import matplotlib.pyplot as plt
import pyvista as pv
from ._3d_vis_utils import _initialize_plotter, _wrap_array
from ..__init__ import timer


@timer
def plot_orthogonal_slices(data, fig: pv.DataSet = None, show_slices: list = None, plotter_kwargs: dict = None,
                           fig_kwargs: dict = None) -> pv.Plotter:
    """
    Input: NumPy array of image, x-, y-, and z-slices to plot
    Output: PyVista plot of orthogonal slices
    """

    if show_slices is None:
        show_slices = [data.nx // 2, data.ny // 2, data.nz // 2]

    if plotter_kwargs is None:
        plotter_kwargs = {}
    if fig_kwargs is None:
        fig_kwargs = {}

    # Test to make sure user only supplied 3 lengths
    assert len(show_slices) == 3, "Please only specify x-, y-, and z-slices to show"
    x_slice, y_slice, z_slice = show_slices

    # Tests to make sure input slices are within image dimensions
    assert 0 <= x_slice < data.nx, "X-slice value outside image dimensions"
    assert 0 <= y_slice < data.ny, "Y-slice value outside image dimensions"
    assert 0 <= z_slice < data.nz, "Z-slice value outside image dimensions"

    # Initialize plotter object
    if fig is None:
        fig = _initialize_plotter(**plotter_kwargs)

    # Wrap NumPy array to pyvista object
    pv_image_obj = _wrap_array(data.image)

    # Extract 3 orthogonal slices
    slices = pv_image_obj.slice_orthogonal(x=x_slice, y=y_slice, z=z_slice)

    # Add the slices as meshes to the PyVista plotter object
    fig.add_mesh(slices, **fig_kwargs)

    return fig


@timer
def plot_contours(data, fig: pv.Plotter = None, show_isosurface: list = None, mesh_kwargs: dict = None,
                  display_kwargs: dict = None) -> pv.Plotter:
    """
    Input: Wrapped image as PyVista object
    Output: Contour at specified isosurface
    """
    if show_isosurface is None:
        show_isosurface = [0.5]

    if mesh_kwargs is None:
        mesh_kwargs = {'opacity': 0.15,
                       'smooth_shading': True,
                       'diffuse': 0.75,
                       'color': (77 / 255, 195 / 255, 255 / 255),
                       'ambient': 0.15}

    if display_kwargs is None:
        display_kwargs = {'filename': data.basename,
                          'take_screenshot': False,
                          'interactive': False}

    if fig is None:
        fig = _initialize_plotter()

    pv_image_obj = _wrap_array(data.image)

    contours = pv_image_obj.contour(isosurfaces=show_isosurface)

    fig.add_mesh(contours, **mesh_kwargs)

    return fig


@timer
def bounding_box(data, fig: pv.Plotter = None, mesh_kwargs: dict = None) -> pv.Plotter:
    """
    Returns wall contours around entire image
    """
    if fig is None:
        fig = _initialize_plotter()

    if mesh_kwargs is None:
        mesh_kwargs = {'opacity': 0.2,
                       'color': (1, 1, 1)}

    wall_bin = data.image.copy()
    wall_bin[1:-1, 1:-1, 1:-1] = 255
    vtk_wall = _wrap_array(wall_bin)
    wall_contours = vtk_wall.contour([255])
    fig.add_mesh(wall_contours, **mesh_kwargs)

    return fig
