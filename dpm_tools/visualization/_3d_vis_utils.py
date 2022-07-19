import numpy as np
import pyvista as pv

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

def _show_3d(plotter_obj, filepath="", take_screenshot=False, interactive=False, **kwargs):

    if take_screenshot:
        cpos = plotter_obj.show(interactive=interactive, return_cpos=True,
                           screenshot=filepath, **kwargs)
        print(cpos)

    else:
        cpos = plotter_obj.show(interactive=True, return_cpos=True)
        print(cpos)



