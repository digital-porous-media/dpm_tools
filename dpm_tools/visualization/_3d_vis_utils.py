import numpy as np
import pyvista as pv
from matplotlib import cm
from matplotlib.colors import ListedColormap
from typing import Tuple
import pathlib


def _initialize_plotter(bg: str = 'w', *args, **kwargs) -> pv.Plotter:
    """
    A helper function to initialize a PyVista Plotter object
    :param bg: The background color to set the plotter to. Defaults to white.
    :returns: A PyVista Plotter object with specified bg and kwargs.
    :rtype: pyvista.Plotter
    """
    # Initialize PV object
    plotter_obj = pv.Plotter(*args, **kwargs)

    # Set background colors
    plotter_obj.set_background(color=bg, **kwargs)

    # Set font colors and sizes
    pv.global_theme.font.color = 'black'
    pv.global_theme.font.size = 18
    pv.global_theme.font.label_size = 14

    return plotter_obj


def _wrap_array(img: np.ndarray) -> pv.DataSet:
    """
    A helper function to wrap a NumPy array to a PyVista object.
    :param img: The NumPy array to wrap.
    :returns: A PyVista DataSet object with the wrapped array.
    :rtype: pyvista.DataSet
    """
    return pv.wrap(img)


def _custom_cmap(vector, color_map: str = 'turbo') -> Tuple[ListedColormap, float, float]:
    """
    Compute a custom color map based on the vector magnitude.
    :param vector: A NumPy array containing the vector data.
    :param color_map: The color map to use. Defaults to 'turbo'.
    :returns: A PyVista colormap object and the minimum and maximum vector magnitude.
    :rtype: (pyvista.Colormap, float, float)
    """
    vector_magnitude = np.sqrt(np.einsum('ij,ij->i', vector, vector))
    log_mag = np.log10(vector_magnitude[vector_magnitude != 0])

    min_magnitude = np.percentile(log_mag, 25)
    max_magnitude = np.percentile(log_mag, 99)
    # print(f'Log min. = {min_magnitude}, Log max. = {max_magnitude}')

    cmap_modified = cm.get_cmap(color_map, 65535)
    spacing = lambda x: np.log10(x)
    new_cmap = ListedColormap(cmap_modified(spacing(np.linspace(1, 10, 65535))))
    # return min_magnitude, max_magnitude
    return new_cmap, 10 ** min_magnitude, 10 ** max_magnitude


def _show_3d(plotter_obj: pv.Plotter, filepath: pathlib.Path, take_screenshot: bool = False, interactive: bool = False,
             **kwargs) -> None:
    """
    A helper function to show a 3D plot with the option to take a screenshot
    :param plotter_obj: The PyVista Plotter object to show.
    :param filepath: The filepath to save the gif to.
    :param take_screenshot: If ``take_screenshot`` is ``True``, this function will take a screenshot of the plot.
    :param interactive: The function shows the plot interactively by default. This can lead to issues when taking a
    screenshot. So we can set it to ``False`` if you don't want to show the plot interactively when taking a screenshot.
    :returns: None
    """
    if take_screenshot:
        cpos = plotter_obj.show(interactive=interactive, return_cpos=True,
                                screenshot=filepath, **kwargs)
        print(cpos)

    else:
        cpos = plotter_obj.show(interactive=True, return_cpos=True)
        print(cpos)


def _initialize_kwargs(plotter_kwargs: dict = None, mesh_kwargs: dict = None) -> Tuple[dict, dict]:
    """
    Utility function to initialize default kwargs for PyVista plotting
    :param plotter_kwargs: A dictionary of kwargs to pass to the PyVista Plotter object.
    :param mesh_kwargs: A dictionary of kwargs to pass to the PyVista Mesh object.
    :returns: Dictionaries of default kwargs to pass to the PyVista Plotter object and PyVista
    :rtype: Tuple[dict, dict]
    """
    if plotter_kwargs is None:
        plotter_kwargs = {}

    if mesh_kwargs is None:
        mesh_kwargs = {'opacity': 0.15,
                       'smooth_shading': True,
                       'diffuse': 0.75,
                       'color': (77 / 255, 195 / 255, 255 / 255),
                       'ambient': 0.15}

    return plotter_kwargs, mesh_kwargs
