r"""

Segmentation
###########################################################################

This submodule contains functions for 2D and 3D segmentation. So far, we have implemented the statistical region merging and seeded region growing algorithms, similar to ImageJ/Fiji.

These are optional dependencies, and the functions will only be available if dpm_srg (for seeded region growing) and dpm_srm (for statistical region merging) are installed.

.. currentmodule:: dpm_tools.segmentation
.. autosummary::
   :template: base_tmpl.rst
   :toctree:

    statistical_region_merging
    seeded_region_growing
"""

from ._segment import statistical_region_merging, seeded_region_growing