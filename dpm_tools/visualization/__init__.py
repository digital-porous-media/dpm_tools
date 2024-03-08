r"""

2D/3D Visualization
###########################################################################

This submodule contains functions for 2D and 3D visualization. 3D visualizations are implemented using PyVista.

.. currentmodule:: dpm_tools.visualization

.. autosummary::
   :template: base_tmpl.rst
   :toctree:

    hist
    plot_slice
    make_thumbnail
    make_gif
    plot_heterogeneity_curve
    orthogonal_slices
    plot_isosurface
    bounding_box
    plot_glyph
    plot_streamlines
    plot_scalar_volume
    plot_medial_axis
"""

from ._plot_2d import *

from ._plot_3d import *

from ._vis_utils import *

from ._3d_vis_utils import *
