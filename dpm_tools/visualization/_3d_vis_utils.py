import numpy as np
import pyvista as pv
from matplotlib import cm
from matplotlib.colors import ListedColormap

def _initialize_plotter(bg: str = 'w', *args, **kwargs):
    """
    A helper function to initialize a PyVista Plotter object
    Default background set to white
    Returns a plotter object
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
    return pv.wrap(img)


def _custom_cmap(vector, color_map: str = 'turbo'):
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


def _show_3d(plotter_obj, filepath="", take_screenshot=False, interactive=False, **kwargs):

    if take_screenshot:
        cpos = plotter_obj.show(interactive=interactive, return_cpos=True,
                           screenshot=filepath, **kwargs)
        print(cpos)

    else:
        cpos = plotter_obj.show(interactive=True, return_cpos=True)
        print(cpos)



