[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.16987906.svg)](https://doi.org/10.5281/zenodo.16987906)
[![image](https://img.shields.io/pypi/v/dpm-tools.svg)](https://pypi.org/project/dpm-tools/)
[![Tests](https://github.com/digital-porous-media/dpm_tools/actions/workflows/tests.yml/badge.svg?branch=master)](https://github.com/digital-porous-media/dpm_tools/actions/workflows/tests.yml)
[![Docs](https://github.com/digital-porous-media/dpm_tools/actions/workflows/docs.yml/badge.svg?branch=master)](https://github.com/digital-porous-media/dpm_tools/actions/workflows/docs.yml)

# Digital Porous Media (DPM) Tools
Welcome to DPM Tools! Our team at The University of Texas at Austin has put together a collection of Python modules for processing and visualizing porous media images. Integrated as workflows, the combination of all module functions provide end-to-end capabilities that streamline pre- and post- processing of porous media images. Individual and combined functions can be used to curate images for publication, prepare images for simulation and machine learning (ML) applications, and analyze simulation and ML results. 

By design, DPM Tools integrate functions from open Python packages to implement common digital porous media as workflows.

DPM Tools is implemented within the Digital Porous Media Portal (formerly Digital Rocks Portal) and as a standalone image processing workflow that users can download.

Current Modules:
---

- ``Input/Output (IO)``: Tools for reading and writing digital porous media images from raw, tiff, hdf5, netCDF and matlab files. The tools can edit and convert the files. 

- ``Metrics``: includes several functions for quantifying geometric properties of images.

- ``Segmentation``: algorithms for classifying phases within an image.

- ``Visualization``: workflows for 3D visualization with PyVista.

Further documentation and descriptions of how to use the different modules are available in the related links of this document and in the landing page where the software is published in Zenodo. Our team is continuously maintaining the software and adding new modules. 


## Installation
To install from PyPI:

    pip install dpm-tools

## Documentation:
For further information and examples on how to get started, please see our [documentation](https://digital-porous-media.github.io/dpm_tools/html/)
