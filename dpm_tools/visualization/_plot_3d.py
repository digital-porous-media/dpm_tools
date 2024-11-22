import numpy as np
import pyvista as pv
from ._3d_vis_utils import _initialize_plotter, _wrap_array, _custom_cmap
import warnings
import skimage


def orthogonal_slices(data, fig: pv.DataSet = None, show_slices: list = None, plotter_kwargs: dict = None,
                      mesh_kwargs: dict = None, slider: bool = False) -> pv.Plotter:
    """
    Plots 3 orthogonal slices of a 3D image.

    Parameters:
        data: A dataclass containing 3D image data
        fig: Pyvista plotter object
        show_slices: List of slices in x, y, z to show. Default is middle slice in each direction.
        plotter_kwargs: Additional keyword arguments to pass to the plotter.
        mesh_kwargs: Pyvista mesh keyword arguments to pass to the plotter.

    Returns:
        pv.Plotter: PyVista plotter object with added orthogonal slice mesh.
    """

    if show_slices is None:
        show_slices = [data.nx // 2, data.ny // 2, data.nz // 2]

    # Overriding the above line because it prevents orthogonal slices from showing for some reason.
    if plotter_kwargs is None:
        plotter_kwargs = {}
    if mesh_kwargs is None:
        mesh_kwargs = {}

    # Test to make sure user only supplied 3 lengths
    assert len(
        show_slices) == 3, "Please only specify x-, y-, and z-slices to show"
    x_slice, y_slice, z_slice = show_slices

    # Tests to make sure input slices are within image dimensions
    assert 0 <= x_slice < data.nx, "X-slice value outside image dimensions"
    assert 0 <= y_slice < data.ny, "Y-slice value outside image dimensions"
    assert 0 <= z_slice < data.nz, "Z-slice value outside image dimensions"

    # Initialize plotter object
    if fig is None:
        fig = _initialize_plotter(**plotter_kwargs)

    # Swapping axes for pyvista compatibility
    ax_swap_arr = np.swapaxes(data.scalar, 0, 2)

    # Wrap NumPy array to pyvista object
    pv_image_obj = _wrap_array(ax_swap_arr)

    # Adding the slider
    if slider is True:
        class MyCustomRoutine:
            def __init__(self, mesh):
                self.output = mesh  # Expected PyVista mesh type
                # default parameters
                self.kwargs = {
                    'z': 1,
                    'x': 1,
                    'y': 1,
                }

            def __call__(self, param, value):
                self.kwargs[param] = int(value)
                self.update()

            def update(self):
                pv_image_obj = _wrap_array(ax_swap_arr)
                result = pv_image_obj.slice_orthogonal(
                    **self.kwargs, contour=True)
                fig.add_mesh(result, name='timestep_mesh', **
                             mesh_kwargs, show_scalar_bar=False)
                self.output.copy_from(result)
                return

        starting_mesh = pv_image_obj.slice_orthogonal(
            x=int(50), y=int(50), z=int(50), contour=True)
        engine = MyCustomRoutine(starting_mesh)
        fig.add_mesh(starting_mesh, name='timestep_mesh',
                     **mesh_kwargs, show_scalar_bar=False)
        _ = fig.add_scalar_bar(
            position_x=0.9, position_y=0.2, height=0.5, vertical=True)
        fig.add_slider_widget(callback=lambda value: engine('z', int(value)),
                              rng=[1, data.nz-1],
                              value=50,
                              pointa=(0.025, 0.1),
                              pointb=(0.31, 0.1),
                              title='Z-slice',
                              fmt="%0.f",
                              style='modern')
        fig.add_slider_widget(callback=lambda value: engine('x', int(value)),
                              rng=[1, data.nx-1],
                              value=50,
                              pointa=(0.35, 0.1),
                              pointb=(0.64, 0.1),
                              title='X-slice',
                              fmt="%0.f",
                              style='modern')
        fig.add_slider_widget(callback=lambda value: engine('y', int(value)),
                              rng=[1, data.ny-1],
                              value=50,
                              pointa=(0.67, 0.1),
                              pointb=(0.98, 0.1),
                              title='Y-slice',
                              fmt="%0.f",
                              style='modern')
        # Only one slider case:
        # def slices_slider(value):
        #     z_slider = int(value)
        #     pv_image_obj = _wrap_array(ax_swap_arr)
        #     slices = pv_image_obj.slice_orthogonal(x=50, y=50, z=z_slider,contour=True)
        #     fig.add_mesh(slices,name='timestep_mesh')
        #     return
        # fig.add_slider_widget(slices_slider, [1, data.nz-1], title='Z-slice',fmt="%0.f")
    else:

        # Extract 3 orthogonal slices
        slices = pv_image_obj.slice_orthogonal(x=x_slice, y=y_slice, z=z_slice)

        # Add the slices as meshes to the PyVista plotter object
        fig.add_mesh(slices, **mesh_kwargs)

    _ = fig.add_axes(
        viewport=(0, 0.8, 0.2, 1),
        line_width=5,
        cone_radius=0.6,
        shaft_length=0.7,
        tip_length=0.3,
        ambient=0.5,
        label_size=(0.4, 0.16),
    )
    return fig


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
        pv.Plotter: PyVista plotter object with added orthogonal slice mesh.
    """

    if mesh_kwargs is None:
        mesh_kwargs = {'opacity': 0.45,
                       'smooth_shading': True,
                       'diffuse': 0.75,
                       'color': (77 / 255, 195 / 255, 255 / 255),
                       'ambient': 0.15}

    if plotter_kwargs is None:
        plotter_kwargs = {}

    if fig is None:
        fig = _initialize_plotter(**plotter_kwargs)

    pv_image_obj = _wrap_array(data.scalar)

    if show_isosurface is None:
        show_isosurface = [(np.amax(data.scalar)+np.amin(data.scalar))/2]
        warnings.warn('\n\nNo value provided for \'show_isosurfaces\' keyword.' +
                      f'Using the midpoint of the isosurface array instead ({np.amin(data.scalar)},{
                          np.amax(data.scalar)}).\n',
                      stacklevel=2)

    contours = pv_image_obj.contour(isosurfaces=show_isosurface)

    fig.add_mesh(contours, **mesh_kwargs)

    return fig


def bounding_box(data, fig: pv.Plotter = None, mesh_kwargs: dict = None, plotter_kwargs: dict = None) -> pv.Plotter:
    """
    Add a bounding box mesh to the Plotter. Assumes the isosurface is at 255.

    Parameters:
        data: A dataclass containing 3D labeled image data
        fig: Pyvista plotter object
        mesh_kwargs: Pyvista mesh keyword arguments to pass to the plotter.
    Returns:
        pv.Plotter: Pyvista plotter object with wall contours around entire image added as a mesh
    """
    if plotter_kwargs is None:
        plotter_kwargs = {}

    if fig is None:
        fig = _initialize_plotter(**plotter_kwargs)

    if mesh_kwargs is None:
        mesh_kwargs = {'opacity': 0.2,
                       'color': (1, 1, 1)}

    wall_bin = data.scalar.copy()
    wall_bin[1:-1, 1:-1, 1:-1] = 255
    vtk_wall = _wrap_array(wall_bin)
    wall_contours = vtk_wall.contour([255])
    fig.add_mesh(wall_contours, **mesh_kwargs)

    return fig


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
        pv.Plotter: Plotter object with glyphs added as a mesh
    """
    if glyph is None:
        glyph = pv.Arrow()

    array = vector_data.magnitude[::glyph_space, ::glyph_space, ::glyph_space].ravel(
    )/np.max(vector_data.magnitude)
    array2 = np.sqrt(array)
    scale_factor = 20
    if glyph_kwargs is None:
        glyph_kwargs = {'scale': array2,
                        'orient': True,
                        'tolerance': 0.05,
                        'geom': glyph,
                        'factor': scale_factor}

    if vector_data.vector is not None:
        glyph_kwargs['orient'] = [vector_data.vector[i][::glyph_space, ::glyph_space,
                                                        ::glyph_space]/np.max(vector_data.magnitude) for i in range(3)]

    x, y, z = np.mgrid[:vector_data.nx:glyph_space,
                       :vector_data.ny:glyph_space,
                       :vector_data.nz:glyph_space]

    # Pseudo mesh for scale bar of the figure
    glyph_kwargs2 = {'scale': array*np.max(vector_data.magnitude),
                     'orient': True,
                     'tolerance': 0.05,
                     'geom': glyph,
                     'factor': scale_factor}
    glyph_kwargs2['orient'] = [vector_data.vector[i][::glyph_space, ::glyph_space,
                                                     ::glyph_space]/np.max(vector_data.magnitude) for i in range(3)]

    fig2 = _initialize_plotter()
    mesh2 = pv.StructuredGrid(z, y, x)
    mesh2['scalars'] = array*np.max(vector_data.magnitude)
    mesh2['vectors'] = np.column_stack((glyph_kwargs2['orient'][0].ravel(),
                                       glyph_kwargs2['orient'][1].ravel(),
                                       glyph_kwargs2['orient'][2].ravel()))
    [glyph_kwargs2.pop(pop_key) for pop_key in ['scale', 'orient']]
    glyphs2 = mesh2.glyph(orient='vectors', scale='scalars', **glyph_kwargs2)
    fig2.add_mesh(glyphs2)

    if plotter_kwargs is None:
        plotter_kwargs = {}
    if mesh_kwargs is None:
        mesh_kwargs = {}

    # Initialize a new plotter object
    if fig is None:
        fig = _initialize_plotter(**plotter_kwargs)

    # Create a structured grid mesh
    mesh = pv.StructuredGrid(z, y, x)
    mesh['scalars'] = array2
    mesh['vectors'] = np.column_stack((glyph_kwargs['orient'][0].ravel(),
                                       glyph_kwargs['orient'][1].ravel(),
                                       glyph_kwargs['orient'][2].ravel()))
    [glyph_kwargs.pop(pop_key) for pop_key in ['scale', 'orient']]
    glyphs = mesh.glyph(orient='vectors', scale='scalars', **glyph_kwargs)
    sargs = dict(mapper=fig2.mapper, height=0.5, width=0.08, vertical=True, position_x=0.10, position_y=0.25,
                 font_family='arial', title_font_size=20, label_font_size=16, fmt="%.2e",
                 title="Magnitude")
    fig.add_mesh(glyphs, scalar_bar_args=sargs, **mesh_kwargs)

    return fig


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
        pv.Plotter: Plotter object with streamlines added as a mesh
    """

    if plotter_kwargs is None:
        plotter_kwargs = {}
    if mesh_kwargs is None:
        mesh_kwargs = {}

    # Initialize a new plotter object if none are provided
    if fig is None:
        fig = _initialize_plotter(**plotter_kwargs)

    mesh = pv.ImageData(dimensions=(vector_data.nz, vector_data.ny, vector_data.nx),
                        spacing=(1.0, 1.0, 1.0),
                        origin=(0.0, 0.0, 0.0))

    x = mesh.points[:, 0]
    y = mesh.points[:, 1]
    z = mesh.points[:, 2]
    vectors = np.array([vector_data.vector[0].flatten('F'),
                        vector_data.vector[1].flatten('F'),
                        vector_data.vector[2].flatten('F')]).T
    mesh['Magnitude'] = vectors

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


def plot_scalar_volume(data, fig: pv.Plotter = None, mesh_kwargs: dict = None,
                       plotter_kwargs: dict = None) -> pv.Plotter:
    """
    Plot voxelized surface to the Plotter object

    Parameters:
        data: A dataclass containing 3D labeled image data
        fig: Pyvista plotter object
        show_isosurface: PyVista keyword arguments to customize the isosurface
        mesh_kwargs: Pyvista mesh keyword arguments to pass to the plotter.
        plotter_kwargs: Additional keyword arguments to pass to the plotter.

    Returns:
        pv.Plotter: Plotter object with voxelized surface added as a volume
    """

    if mesh_kwargs is None:
        mesh_kwargs = {}

    if plotter_kwargs is None:
        plotter_kwargs = {}

    if fig is None:
        fig = _initialize_plotter(**plotter_kwargs)

    # Create a bounded volume
    # wall_bin = 255 * np.ones((data.scalar.shape[0]+2, data.scalar.shape[1]+2, data.scalar.shape[2]+2))
    # wall_bin[1:-1, 1:-1, 1:-1] = data.scalar.copy()

    mesh = pv.ImageData(dimensions=(data.nz, data.ny, data.nx),
                        spacing=(1.0, 1.0, 1.0),
                        origin=(0.0, 0.0, 0.0))

    mesh['scalars'] = data.scalar.flatten(order="F")

    data.scalar[data.scalar == 0.0] = np.nan

    fig.add_volume(mesh, opacity='foreground', **mesh_kwargs)

    return fig


def plot_medial_axis(data, fig: pv.Plotter = None, show_isosurface: list = None,
                     mesh_kwargs: dict = None, plotter_kwargs: dict = None, notebook=False) -> pv.Plotter:
    """
    Plots an interactive visual with a medial axis and a 3D isosurface of given data.

    Parameters:
        data: A dataclass containing 3D labeled image data
        fig: Pyvista plotter object
        show_isosurface: List of isosurfaces to show. Default is single isosurface at average between maximum and minimum label values.
        notebook: True for rendring in Jupyter notebook. Defaults to False.
        mesh_kwargs: Pyvista mesh keyword arguments to pass to the plotter.
        plotter_kwargs: Additional keyword arguments to pass to the plotter. Defaults to None.
    Returns:
        pv.Plotter: PyVista plotter object with added orthogonal slice mesh.
    """

    # plotter_kwargs, mesh_kwargs = _initialize_kwargs(plotter_kwargs, mesh_kwargs)
    if mesh_kwargs is None:
        mesh_kwargs = {'opacity': 0.45,
                       'smooth_shading': True,
                       'diffuse': 0.75,
                       'color': (77 / 255, 195 / 255, 255 / 255),
                       'ambient': 0.15}

    if plotter_kwargs is None:
        plotter_kwargs = {}

    if fig is None:
        fig = _initialize_plotter(**plotter_kwargs)

    medial_axis = skimage.morphology.skeletonize(data.scalar)
    pv_image_obj = _wrap_array(medial_axis)

    contours_ma = pv_image_obj.contour(isosurfaces=[0.5])
    fig.add_mesh(contours_ma, style='wireframe', color='r',
                 line_width=2, name='medial_axis')

    fig = plot_isosurface(data, fig=fig, mesh_kwargs=mesh_kwargs)

    return fig
