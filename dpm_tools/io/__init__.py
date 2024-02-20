r"""

Input/Output functions for reading and writing images.
###########################################################################

This module contains functions for reading and writing 3D volumetric images from a variety of file formats.

.. currentmodule:: dpm_tools.io

.. autosummary::
   :template: base_tmpl.rst
   :toctree:

    read_image
    write_image
    ImageFromFile
    Image
    Vector

"""

from ._io_utils import *

from ._read_data import *

from ._write_data import *
