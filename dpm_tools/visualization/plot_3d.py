import numpy as np
import matplotlib.pyplot as plt
import pyvista as pv
from ._3d_vis_utils import _initialize_plotter, _wrap_array, _custom_cmap, _initialize_kwargs
from ..__init__ import timer
import warnings


@timer
def orthogonal_slices(data, fig: pv.DataSet = None, show_slices: list = None, plotter_kwargs: dict = None,
                      mesh_kwargs: dict = None) -> pv.Plotter:
    """
    Plots 3 orthogonal slices of a 3D image.
    Parameters:
        data: A dataclass containing 3D image data
        fig: Pyvista plotter object
        show_slices: List of slices in x, y, z to show. Default is middle slice in each direction.
        plotter_kwargs: Additional keyword arguments to pass to the plotter.
        mesh_kwargs: Pyvista mesh keyword arguments to pass to the plotter.
    Returns:
        fig: PyVista plotter object with added orthogonal slice mesh.
    """

    if show_slices is None:
        show_slices = [data.nx // 2, data.ny // 2, data.nz // 2]

    plotter_kwargs, mesh_kwargs = _initialize_kwargs(plotter_kwargs, mesh_kwargs)

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
    fig.add_mesh(slices, **mesh_kwargs)

    return fig


@timer
def plot_isosurface(data, fig: pv.Plotter = None, show_isosurface: list = None, mesh_kwargs: dict = None,
                    plotter_kwargs: dict = None) -> pv.Plotter:
    """
    Plots 3D isosurfaces
    Parameters:
        data: A dataclass containing 3D labeled image data
        fig: Pyvista plotter object
        show_isosurface: List of isosurfaces to show. Default is single isosurface at average between maximum and minimum label values.
        mesh_kwargs: Pyvista mesh keyword arguments to pass to the plotter.
        plotter_kwargs: Additional keyword arguments to pass to the plotter. Defaults to None.
    Returns:
        fig: PyVista plotter object with added orthogonal slice mesh.
    """

    plotter_kwargs, mesh_kwargs = _initialize_kwargs(plotter_kwargs, mesh_kwargs)

    if fig is None:
        fig = _initialize_plotter(**plotter_kwargs)
    
    pv_image_obj = _wrap_array(data.image)

    if show_isosurface is None:
        show_isosurface = [(np.amax(data.image)+np.amin(data.image))/2]
        warnings.warn(f"No isosurfaces specified. Using isosurfaces {show_isosurface}")

    contours = pv_image_obj.contour(isosurfaces=show_isosurface)
    
    fig.add_mesh(contours, **mesh_kwargs)

    return fig


@timer
def bounding_box(data, fig: pv.Plotter = None, mesh_kwargs: dict = None, plotter_kwargs: dict = None) -> pv.Plotter:
    """
    Add a bounding box mesh to the Plotter. Assumes the isosurface is at 255.

    Parameters:
        data: A dataclass containing 3D labeled image data
        fig: Pyvista plotter object
        mesh_kwargs: Pyvista mesh keyword arguments to pass to the plotter.
    Returns:
        Pyvista plotter object with wall contours around entire image
    """

    if fig is None:
        fig = _initialize_plotter(**plotter_kwargs)

    plotter_kwargs, mesh_kwargs = _initialize_kwargs(plotter_kwargs, mesh_kwargs)

    wall_bin = data.image.copy()
    wall_bin[1:-1, 1:-1, 1:-1] = 255
    vtk_wall = _wrap_array(wall_bin)
    wall_contours = vtk_wall.contour([255])
    fig.add_mesh(wall_contours, **mesh_kwargs)

    return fig

@timer
def plot_glyph(vector_data, fig: pv.Plotter = None, glyph: pv.PolyData = None, glyph_space: int = 1,
               glyph_kwargs: dict = None, mesh_kwargs: dict = None, plotter_kwargs: dict = None) -> pv.Plotter:
    """
    Plot glyphs to the Plotter such as arrows, spheres, etc. for vector fields

    Parameters:
        vector_data: A dataclass containing 3D vector data in 3 component directions
        fig: Pyvista plotter object
        glyph: PyVista polydata object to add to the plotter. Defaults to arrow glyph
        glyph_space: Spacing between glyphs. Defaults to 1
        glyph_kwargs: Additional keyword arguments to customize the glyph
        mesh_kwargs: Pyvista mesh keyword arguments to pass to the plotter.
        plotter_kwargs: Additional keyword arguments to pass to the plotter.
    Returns:
        Pyvista plotter object with glyph object
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

    plotter_kwargs, mesh_kwargs = _initialize_kwargs(plotter_kwargs, mesh_kwargs)

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
               streamline_kwargs: dict = None, mesh_kwargs: dict = None, plotter_kwargs: dict = None) -> pv.Plotter:
    """
    Plot streamlines to the Plotter object

    Parameters:
        vector_data: A dataclass containing 3D vector data in 3 component directions
        fig: Pyvista plotter object
        tube_radius: Radius of streamline tube. Defaults to 0.75/224 * vector_data.nz
        streamline_kwargs: PyVista keyword arguments to customize the streamline
        mesh_kwargs: Pyvista mesh keyword arguments to pass to the plotter.
        plotter_kwargs: Additional keyword arguments to pass to the plotter.
    Returns:
        Pyvista plotter object with glyph object
    """

    plotter_kwargs, _ = _initialize_kwargs(plotter_kwargs, mesh_kwargs)

    # Initialize a new plotter object if none are provided
    if fig is None:
        fig = _initialize_plotter(**plotter_kwargs)

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

