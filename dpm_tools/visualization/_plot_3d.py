import numpy as np
import pyvista as pv
from dpm_tools.io import Image
from ._3d_vis_utils import _initialize_plotter, _wrap_array, _custom_cmap
import warnings
import skimage
from copy import deepcopy
import cc3d
import scipy.stats
import inspect
from tqdm.notebook import tqdm

def orthogonal_slices(data, fig: pv.DataSet = None, show_slices: list = None, plotter_kwargs: dict = None,
                      mesh_kwargs: dict = None, slider: bool = False) -> pv.Plotter:
    """
    Plots 3 orthogonal slices of a 3D image.

    Parameters:
        data: A np array containing 3D image data
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
                fig.add_mesh(result, name='timestep_mesh',
                             **mesh_kwargs, show_scalar_bar=False)
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
                              rng=[1, data.nz - 1],
                              value=50,
                              pointa=(0.025, 0.1),
                              pointb=(0.31, 0.1),
                              title='Z-slice',
                              fmt="%0.f",
                              style='modern')
        fig.add_slider_widget(callback=lambda value: engine('x', int(value)),
                              rng=[1, data.nx - 1],
                              value=50,
                              pointa=(0.35, 0.1),
                              pointb=(0.64, 0.1),
                              title='X-slice',
                              fmt="%0.f",
                              style='modern')
        fig.add_slider_widget(callback=lambda value: engine('y', int(value)),
                              rng=[1, data.ny - 1],
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


def plot_isosurface(data, fig: pv.Plotter = None, show_isosurface: list = None, smooth_kwargs: dict = None, mesh_kwargs: dict = None,
                    plotter_kwargs: dict = None) -> pv.Plotter:
    """
    Plots 3D isosurfaces

    Parameters:
        data: A dataclass containing 3D labeled image data
        fig: Pyvista plotter object
        show_isosurface: List of isosurfaces to show. Default is single isosurface at average between maximum and minimum label values.
        smooth_kwargs: Pyvista smoothing keyword arguments from [pyvista.PolyDataFilters.smooth](https://docs.pyvista.org/api/core/_autosummary/pyvista.polydatafilters.smooth#pyvista.PolyDataFilters.smooth). Defaults to None = No surface smoothing.
        mesh_kwargs: Pyvista mesh keyword arguments to pass to the plotter. Defaults to None.
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
        show_isosurface = [(np.amax(data.scalar) + np.amin(data.scalar)) / 2]
        warnings.warn("\n\nNo value provided for \'show_isosurfaces\' keyword. " +
                      f"Using the midpoint of the isosurface array instead ({np.amin(data.scalar)},\
                      {np.amax(data.scalar)}).\n",
                      stacklevel=2)
    
    contours = pv_image_obj.contour(isosurfaces=show_isosurface)
    if smooth_kwargs is not None:
        contours = contours.smooth_taubin(**smooth_kwargs)

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

    array = vector_data.magnitude[::glyph_space,
                                  ::glyph_space,
                                  ::glyph_space].ravel() / np.max(vector_data.magnitude)
    array2 = np.sqrt(array)
    scale_factor = 20
    if glyph_kwargs is None:
        glyph_kwargs = {'scale': array2,
                        'orient': True,
                        'tolerance': 0.05,
                        'geom': glyph,
                        'factor': scale_factor}

    if vector_data.vector is not None:
        glyph_kwargs['orient'] = [vector_data.vector[i][::glyph_space,
                                                        ::glyph_space,
                                                        ::glyph_space] / np.max(vector_data.magnitude) for i in range(3)]

    # plotter_kwargs, mesh_kwargs = _initialize_kwargs(plotter_kwargs, mesh_kwargs)
    x, y, z = np.mgrid[:vector_data.nx:glyph_space,
                       :vector_data.ny:glyph_space,
                       :vector_data.nz:glyph_space]

    # Pseudo mesh for scale bar of the figure
    glyph_kwargs2 = {'scale': array * np.max(vector_data.magnitude),
                     'orient': True,
                     'tolerance': 0.05,
                     'geom': glyph,
                     'factor': scale_factor}
    glyph_kwargs2['orient'] = [vector_data.vector[i][::glyph_space, ::glyph_space,
                                                     ::glyph_space] / np.max(vector_data.magnitude) for i in range(3)]

    fig2 = _initialize_plotter()
    mesh2 = pv.StructuredGrid(z, y, x)
    mesh2['scalars'] = array * np.max(vector_data.magnitude)
    mesh2['vectors'] = np.column_stack((glyph_kwargs2['orient'][0].ravel(),
                                        glyph_kwargs2['orient'][1].ravel(),
                                        glyph_kwargs2['orient'][2].ravel()))
    [glyph_kwargs2.pop(pop_key) for pop_key in ['scale', 'orient']]
    glyphs2 = mesh2.glyph(
        orient='vectors', scale='scalars', **glyph_kwargs2)
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
        tube_radius = 0.75 / 224 * vector_data.nz

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

    # data.scalar[data.scalar == 0.0] = np.nan

    fig.add_volume(mesh, **mesh_kwargs)

    return fig


def extract_competent_subset(data: np.ndarray, 
                             cube_size: int = 100, 
                             batch: int = 100, 
                             pore_class: int = 0, 
                             class_to_optimize: int = 0,
                             for_3d_printing: bool = False,
                             max_overhang_angle = 20,
                             exhaustive_search: bool = False,
                             debug: bool = False,
                             show_progress: bool = True) -> np.ndarray:
    '''
    Finds the best cubic subset for visualizing the segmented dataset.
    Parameters:
        data: 3D numpy array, vector class from DPM Tools, Image class from DPM Tools.
        cube_size: Size of the visualization cube, default is 100 (100x100x100).
        batch: Batch size over which to calculate the stats, default is 100.
        pore_class: The class representing pores in the original data, default is 0.
        class_to_optimize: The class to optimize connectivity for (0 or 1), default is 0.
        for_3d_printing: If True, evaluate printability metrics, default is False.
        max_overhang_angle: Maximum overhang angle for 3D printing evaluation, default is 20.
        exhaustive_search: If True, search entire 3D volume independently. If False, search diagonally (faster), default is False.
        debug: If True, print detailed debugging information, default is False.
        show_progress: If True, display tqdm progress bar, default is True.
    Returns:
        tuple: (best_subset_range, stats_array) where best_subset_range maximizes connected component of class_to_optimize
    '''
    if str(type(data)) == "<class 'dpm_tools.io.read_data.Vector'>":
        scalar_data = deepcopy(data.image)
    elif str(type(data)) == "<class 'dpm_tools.io.read_data.Image'>":
        scalar_data = deepcopy(data)
    else:
        scalar_data = deepcopy(data)
    
    if debug:
        print(f"Input data shape: {scalar_data.shape}, unique values: {np.unique(scalar_data)}")
        print(f"pore_class={pore_class}, class_to_optimize={class_to_optimize}")
    
    # Store original for porosity calculation
    original_data = scalar_data.copy()
    
    # Step 1: Convert class_to_optimize to 1, the other class to 0
    scalar_data = (scalar_data == class_to_optimize).astype(np.uint8)
    
    if debug:
        print(f"After conversion, class {class_to_optimize} is now 1")
        print(f"Fraction of 1s: {scalar_data.sum() / scalar_data.size:.3f}")
    
    size = original_data.shape[0] * original_data.shape[1] * original_data.shape[2]
    
    # Calculate porosity based on pore_class in original data
    porosity = (original_data == pore_class).sum() / size
    
    if debug:
        print(f"Pore fraction (class {pore_class}): {porosity:.3f}")
    
    sample_size = cube_size
    # Inner cube increment - use 25% from each edge
    inc = int(sample_size * 0.25)
    
    if debug:
        print(f"Outer cube: {sample_size}^3, Inner cube: {sample_size - 2*inc}^3")
        print(f"Search mode: {'Exhaustive 3D' if exhaustive_search else 'Diagonal (faster)'}")
    
    if exhaustive_search:
        max_dim_x = scalar_data.shape[0]
        max_dim_y = scalar_data.shape[1]
        max_dim_z = scalar_data.shape[2]
    else:
        max_dim = len(scalar_data)
    
    # Warn if dataset is very large
    if debug:
        if exhaustive_search:
            dataset_voxels = max_dim_x * max_dim_y * max_dim_z
            if dataset_voxels > 1000**3:
                print(f"Warning: Large dataset ({max_dim_x}x{max_dim_y}x{max_dim_z}). This may take several minutes on CPU.")
    
    batch_for_stats = batch
    if for_3d_printing:
        if exhaustive_search:
            stats_array = np.zeros(shape=(8, batch_for_stats))
        else:
            stats_array = np.zeros(shape=(7, batch_for_stats))
    else:
        if exhaustive_search:
            stats_array = np.zeros(shape=(6, batch_for_stats))
        else:
            stats_array = np.zeros(shape=(5, batch_for_stats))
    
    i = 0
    attempts = 0
    max_attempts = batch * 100
    
    # Debug counters
    fail_no_inner = 0
    fail_no_components = 0
    fail_no_center = 0
    fail_porosity = 0
    
    if debug:
        print(f"Starting search for {batch_for_stats} valid cubes...")
    
    # Create progress bar
    if show_progress:
        pbar = tqdm(total=batch_for_stats, 
                   desc="Finding valid subsets",
                   unit="subset",
                   bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]')
    
    while i < batch_for_stats and attempts < max_attempts:
        attempts += 1
        
        if debug and attempts % 1000 == 0:
            print(f"Attempt {attempts}, found {i} valid cubes")
            print(f"  Failures: no_inner={fail_no_inner}, no_comp={fail_no_components}, no_center={fail_no_center}, porosity={fail_porosity}")
        
        if exhaustive_search:
            mini_x = np.random.randint(low=0, high=max_dim_x - sample_size)
            mini_y = np.random.randint(low=0, high=max_dim_y - sample_size)
            mini_z = np.random.randint(low=0, high=max_dim_z - sample_size)
            maxi_x = mini_x + sample_size
            maxi_y = mini_y + sample_size
            maxi_z = mini_z + sample_size
        else:
            mini = np.random.randint(low=0, high=max_dim - sample_size)
            maxi = mini + sample_size
            mini_x = mini_y = mini_z = mini
            maxi_x = maxi_y = maxi_z = maxi
        
        # Extract cube from the converted data (where class_to_optimize is now 1)
        scalar_boot = scalar_data[mini_x:maxi_x, mini_y:maxi_y, mini_z:maxi_z]
        
        # Also get the original cube for porosity checking
        scalar_boot_original = original_data[mini_x:maxi_x, mini_y:maxi_y, mini_z:maxi_z]

        # Check if there's any of class 1 in inner region
        inner_sum = scalar_boot[inc:-inc, inc:-inc, inc:-inc].sum()
        if inner_sum == 0:
            fail_no_inner += 1
            continue
        
        # Get largest connected component using cc3d.largest_k
        labels_outer, N = cc3d.largest_k(
            scalar_boot,
            k=1,
            connectivity=26,
            delta=0,
            return_N=True,
        )
        
        if N == 0:  # No components found
            fail_no_components += 1
            continue
        
        # Find the actual largest component label by counting
        unique_labels, counts = np.unique(labels_outer, return_counts=True)
        
        # Remove background (label 0)
        non_zero_mask = unique_labels > 0
        if non_zero_mask.sum() == 0:
            fail_no_components += 1
            continue
        
        unique_labels = unique_labels[non_zero_mask]
        counts = counts[non_zero_mask]
        
        # Get the label with the largest count
        largest_label = unique_labels[np.argmax(counts)]
        
        # Create mask for largest component
        lcc_mask = (labels_outer == largest_label)
        
        # Check if this component reaches the inner cube
        inner_lcc = lcc_mask[inc:-inc, inc:-inc, inc:-inc]
        inner_count = inner_lcc.sum()
        
        if inner_count == 0:  # Component doesn't reach center
            fail_no_center += 1
            continue
        
        outer_count = lcc_mask.sum()
        
        # Calculate porosity of the pore_class in this cube (from original data)
        porosity_selected = (scalar_boot_original == pore_class).sum() / sample_size**3
        
        # Check if this cube's porosity is within 20% of overall porosity
        if (porosity_selected <= porosity * 1.2) & (porosity_selected >= porosity * 0.8):
            stats_array[0, i] = outer_count
            stats_array[1, i] = inner_count
            stats_array[2, i] = porosity_selected
            stats_array[3, i] = mini_x
            if exhaustive_search:
                stats_array[4, i] = mini_y
                stats_array[5, i] = mini_z
            
            if for_3d_printing:
                volume = np.pad(scalar_boot, ((1, 1), (1, 1), (1, 1)), mode='constant', constant_values=1).copy()
                # -----------------------------
                # Marching cubes (voxel-true surface)
                # -----------------------------
                verts, faces, _, _ = skimage.measure.marching_cubes(volume,level=0.5)
            
                # Convert faces to PyVista format
                faces_pv = np.hstack([np.full((faces.shape[0], 1), 3), faces])
                pv_mesh = pv.PolyData(verts, faces_pv)

                # # -----------------------------
                # # Simplify mesh (OPTIONAL)
                # # -----------------------------
                # if simplify is not None:
                #     if not (0.0 < simplify < 1.0):
                #         raise ValueError("simplify must be between 0 and 1")
                #     pv_mesh = pv_mesh.decimate_pro(simplify, preserve_topology=True)
            
                # -----------------------------
                # Recompute normals AFTER simplification
                # -----------------------------
                pv_mesh.compute_normals(cell_normals=True,point_normals=False,inplace=True)
                face_normals = pv_mesh.cell_normals
            
                # -----------------------------
                # Overhang detection
                # -----------------------------
                build_dir = np.array([0.0, 0.0, 1.0])
                cos_theta = face_normals @ build_dir
                angles = np.degrees(np.arccos(np.clip(cos_theta, -1.0, 1.0)))
            
                overhang_faces = angles > (90.0 + max_overhang_angle)
            
                face_areas = pv_mesh.compute_cell_sizes(length=False, area=True, volume=False)["Area"]
                percent_overhang_area = (100.0 * face_areas[overhang_faces].sum() / face_areas.sum())

                if percent_overhang_area > 30:
                    continue
                connectivity_score = scipy.stats.hmean([outer_count, inner_count])
                printability_penalty = percent_overhang_area / 100.0

                gamma = 4.0  # 2 = mild, 4 = brutal
                score = connectivity_score * (1.0 - printability_penalty) ** gamma

                # score = connectivity_score * (1.0 - printability_penalty)
                                                
                stats_array[6, i] = percent_overhang_area
                stats_array[7, i] = score
            
            i += 1
            
            # Update progress bar
            if show_progress:
                pbar.update(1)
                pbar.set_postfix({'attempts': attempts, 'ratio': f'{i/attempts:.2%}'})

        
            if debug and (i <= 5 or i % 10 == 0):
                print(f"Found {i}/{batch_for_stats} - outer: {outer_count}, inner: {inner_count}, porosity: {porosity_selected:.3f}")
        else:
            fail_porosity += 1
    
    # Close progress bar
    if show_progress:
        pbar.close()
    
    if debug:
        print(f"\nSearch complete after {attempts} attempts")
        print(f"Final failures: no_inner={fail_no_inner}, no_comp={fail_no_components}, no_center={fail_no_center}, porosity={fail_porosity}")
    
    if i == 0:
        raise ValueError(f"Could not find any valid cubes after {attempts} attempts.")
    
    if for_3d_printing:
        if exhaustive_search:
            best_index = np.argmax(stats_array[7, :i])
            print(f"Overhang surface area: {stats_array[6, best_index]:.2f}%")
        else:
            best_index = np.argmax(stats_array[6, :i])
            print(f"Overhang surface area: {stats_array[5, best_index]:.2f}%")
    else:
        if exhaustive_search:
            # For exhaustive non-3D printing, compute harmonic mean here
            hmean_scores = scipy.stats.hmean([stats_array[0, :i], stats_array[1, :i]], axis=0)
            best_index = np.argmax(hmean_scores)
        else:
            best_index = np.argmax(stats_array[4, :i])
    
    best_x = int(stats_array[3, best_index])
    if exhaustive_search:
        best_y = int(stats_array[4, best_index])
        best_z = int(stats_array[5, best_index])
    else:
        best_y = best_x
        best_z = best_x
    
    # Always print these final results
    print(f'Original Porosity (class {pore_class}): {round(porosity * 100, 2)} %')
    print(f'Subset Porosity: {round(stats_array[2, best_index] * 100, 2)} %')
    print(f'Optimized for class {class_to_optimize} connectivity')
    print(f'Competent Subset: [{best_x}:{best_x + cube_size},'+
          f'{best_y}:{best_y + cube_size}, {best_z}:{best_z + cube_size}]')
    
    if exhaustive_search:
        best_subset_range = ((int(best_x), int(best_x + cube_size)),
                             (int(best_y), int(best_y + cube_size)),
                             (int(best_z), int(best_z + cube_size)))
    else:
        best_subset_range = (int(best_x), int(best_x + cube_size))
    
    return best_subset_range, stats_array


def plot_medial_axis(data, fig: pv.Plotter = None, pore_class=0, interactive=False,show_isosurface: list = None,
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
    if isinstance(data, Image):
        data = deepcopy(data.scalar)
    elif inspect.isclass(data):
        data = deepcopy(data.image)
    else:
        data = deepcopy(data)

    if mesh_kwargs is None:
        mesh_kwargs = {'opacity': 0.45,
                       'smooth_shading': True,
                       'diffuse': 0.75,
                       'color': (77 / 255, 195 / 255, 255 / 255),
                       'ambient': 0.15}

    if plotter_kwargs is None:
        plotter_kwargs = {}

    pv.set_jupyter_backend('server')
    if fig is None:
        fig = _initialize_plotter(**plotter_kwargs)

    if pore_class==1:
        inv_data_8bit = np.invert(data)
        inv_data = skimage.exposure.rescale_intensity(inv_data_8bit,out_range=(0,1)).astype('int')
        medial_axis = skimage.morphology.skeletonize(inv_data)
    elif pore_class==0:
        medial_axis = skimage.morphology.skeletonize(data)
    else: 
        raise ValueError('pore_class should either be 0 or 1.')

    pv_image_obj = _wrap_array(medial_axis)

    contours_ma = pv_image_obj.contour(isosurfaces=[0.5])
    fig.add_mesh(contours_ma, style='wireframe', color='r',
                 line_width=2, name='medial_axis')

    pv_obj_sample = _wrap_array(np.pad(data, ((1, 1), (1, 1), (1, 1)), mode='constant', constant_values=1))
    contours_sample = pv_obj_sample.contour(isosurfaces=[0.5])


    def my_plane_func(normal, origin):
        fig.add_mesh(contours_sample.clip_closed_surface(normal='-z', origin=origin),
                    name='arrows',color = (200 / 255, 181 / 255, 152 / 255))

    if interactive == True:
        fig.add_plane_widget(my_plane_func, normal='z',origin=[0, 0, medial_axis.shape[2]+2])
    else:
        fig.add_plane_widget(my_plane_func, normal='z',origin=[0, 0, medial_axis.shape[2]/2])
    
    return fig

