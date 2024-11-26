r"""

Input/Output functions for reading and writing images.
###########################################################################

This module contains functions for reading and writing 3D volumetric images from a variety of file formats.

.. currentmodule:: dpm_tools.io

.. autosummary::
   :template: base_tmpl.rst
   :toctree:

    Image
    read_image
    write_image
    find_files_with_ext
    get_tiff_metadata
    natural_sort
    combine_slices
    convert_filetype
    download_file

"""

from ._io_utils import find_files_with_ext, get_tiff_metadata, natural_sort, combine_slices, convert_filetype, download_file

from ._read_data import read_image, Image

from ._write_data import write_image
