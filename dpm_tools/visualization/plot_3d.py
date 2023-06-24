import numpy as np
import matplotlib.pyplot as plt
import pyvista as pv
from ._3d_vis_utils import _initialize_plotter, _wrap_array, _custom_cmap
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

    if mesh_kwargs is None:
        mesh_kwargs = {'opacity': 0.15,
                       'smooth_shading': True,
                       'diffuse': 0.75,
                       'color': (77 / 255, 195 / 255, 255 / 255),
                       'ambient': 0.15}

    if display_kwargs is None:
        display_kwargs = {}
        # display_kwargs = {'filename': data.basename,
        #                   'take_screenshot': False,
        #                   'interactive': False}

    if fig is None:
        fig = _initialize_plotter()
    
    pv_image_obj = _wrap_array(data.image)

    if show_isosurface is None:
        # show_isosurface = [np.mean(_initialize_plotter().add_mesh(pv_image_obj.contour()).mapper.scalar_range)]
        show_isosurface = [(np.amax(data.image)+np.amin(data.image))/2]
    print(show_isosurface)
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

@timer
def plot_glyph(vector_data, fig: pv.Plotter = None, glyph: pv.PolyData = None, glyph_space: int = 1,
               glyph_kwargs: dict = None, mesh_kwargs: dict = None) -> pv.Plotter:
    """
    Plot glyph images such as spheres, vector fields, etc.
    """
    if glyph is None:
        glyph = pv.Arrow(scale=100)

    if glyph_kwargs is None:
        glyph_kwargs = {'scale': vector_data.magnitude[::glyph_space, ::glyph_space, ::glyph_space].ravel()/np.max(vector_data.magnitude),
                        'orient': True,
                        'tolerance': 0.05,
                        'geom': glyph}

        if vector_data.vector is not None:
            glyph_kwargs['orient'] = [vector_data.vector[i][::glyph_space, ::glyph_space, ::glyph_space]/np.max(vector_data.magnitude) for i in range(3)]


    if mesh_kwargs is None:
        mesh_kwargs = {}

    x, y, z = np.mgrid[:vector_data.nx:glyph_space,
                       :vector_data.ny:glyph_space,
                       :vector_data.nz:glyph_space]

    # Initialize a new plotter object
    if fig is None:
        fig = _initialize_plotter()

    # Create a structured grid mesh
    mesh = pv.StructuredGrid(z, y, x)
    mesh['scalars'] = glyph_kwargs['scale']
    mesh['vectors'] = np.column_stack((glyph_kwargs['orient'][0].ravel(),
                                       glyph_kwargs['orient'][1].ravel(),
                                       glyph_kwargs['orient'][2].ravel()))
    [glyph_kwargs.pop(pop_key) for pop_key in ['scale', 'orient']]
    glyphs = mesh.glyph(orient='vectors', scale='scalars', **glyph_kwargs)
    sargs = dict(height=0.5, width=0.08, vertical=True, position_x=0.10, position_y=0.25,
                 font_family='arial', title_font_size=45, label_font_size=36, fmt="%.2e",
                 title="Magnitude")
    fig.add_mesh(glyphs, scalar_bar_args=sargs, **mesh_kwargs)

    return fig


@timer
def plot_streamlines(vector_data, fig: pv.Plotter = None, tube_radius: float = None,
               streamline_kwargs: dict = None, mesh_kwargs: dict = None) -> pv.Plotter:




    # Initialize a new plotter object
    if fig is None:
        fig = _initialize_plotter()

    mesh = pv.UniformGrid(dims=(vector_data.nz, vector_data.ny, vector_data.nx),
                          spacing=(1.0, 1.0, 1.0),
                          origin=(0.0, 0.0, 0.0))

    mesh['Magnitude'] = np.array([vector_data.vector[0].flatten('F'),
                                  vector_data.vector[1].flatten('F'),
                                  vector_data.vector[2].flatten('F')]).T

    if streamline_kwargs is None:
        streamline_kwargs = {'n_points': int(vector_data.nz**1.25),
                             'source_radius': vector_data.nz // 2,
                             'terminal_speed': 0.0,
                             'initial_step_length': 2.0}

    stream, src = mesh.streamlines('Magnitude', return_source=True,
                                   **streamline_kwargs)

    if tube_radius is None:
        tube_radius = 0.75/224 * vector_data.nz

    if mesh_kwargs is None:
        my_cmap, cmin, cmax = _custom_cmap(mesh['Magnitude'], 'gnuplot')
        mesh_kwargs = {'colormap': my_cmap,
                       'clim': [cmin, cmax]}

    fig.add_mesh(stream.tube(radius=tube_radius), **mesh_kwargs)

    return fig

