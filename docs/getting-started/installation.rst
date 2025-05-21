.. _installation:

####################
Installing DPM Tools
####################

DPM Tools is a collection of tools for analysis of 2D and 3D porous media images. We tried to keep dependencies to a minimum and to only rely on common Python packages. 

For the best experience, we recommend installing DPM Tools in a virtual environment. 

Python version
--------------
DPM Tools requires at least Python 3.8, but **Python 3.10+** is recommended.


Required dependencies
---------------------
DPM Tools requires the following packages:

+-------------------------+---------------------------------------------------------+
| Dependency              | Purpose                                                 |
+=========================+=========================================================+
| numpy                   | Core numerical computations and array operations.       |
+-------------------------+---------------------------------------------------------+
| matplotlib              | Plotting and data visualization.                        |
+-------------------------+---------------------------------------------------------+
| pyvista[all]            | 3D plotting and mesh analysis using VTK under the hood. |
+-------------------------+---------------------------------------------------------+
| pandas                  | Data manipulation and tabular data handling.            |
+-------------------------+---------------------------------------------------------+
| tifffile                | Reading and writing TIFF image files.                   |
+-------------------------+---------------------------------------------------------+
| exifread                | Extracting metadata (EXIF) from image files.            |
+-------------------------+---------------------------------------------------------+
| netcdf4                 | Reading and writing NetCDF files for scientific data.   |
+-------------------------+---------------------------------------------------------+
| h5py                    | Interfacing with HDF5 datasets.                         |
+-------------------------+---------------------------------------------------------+
| porespy                 | Image analysis for porous media.                        |
+-------------------------+---------------------------------------------------------+
| edt                     | Euclidean distance transforms on binary images.         |
+-------------------------+---------------------------------------------------------+
| connected-components-3d | 3D connected component labeling for binary volumes.     |
+-------------------------+---------------------------------------------------------+
| pyfftw                  | Fast Fourier transforms using the FFTW library.         |
+-------------------------+---------------------------------------------------------+
| dpm_srm                 | Statistical region merging segmentation.                |
+-------------------------+---------------------------------------------------------+
| dpm_srg                 | Seeded region growing segmentation.                     |
+-------------------------+---------------------------------------------------------+


Install stable releases
=======================

Installing with Pip:
--------------------

Stable releases can be installed from `PyPI <https://pypi.org/project/dpm-tools>`_ using ``pip``::

   pip install dpm_tools


.. Installing with Conda:
.. ----------------------

.. Alternatively, DPM Tools can be installed with ``conda``

..    conda install -c conda-forge dpm_tools


Installing the Development Branch from GitHub 
---------------------------------------------
You can install the latest version from GitHub by cloning `DPM Tools <https://github.com/digital-porous-media/dpm_tools>`_, and running::
   
   git clone https://github.com/digital-porous-media/dpm_tools.git
   cd dpm_tools
   python -m pip install -e .[all]

This will install DPM Tools and development tools for testing and building documentation in an editable environment.