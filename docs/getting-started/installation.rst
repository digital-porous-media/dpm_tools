.. _installation:

############
Installing DPM Tools
############

DPM Tools is a collection of tools for analysis of 2D and 3D porous media images. We tried to keep dependencies to a minimum and to only rely on common Python packages. 

For the best experience, we recommend installing DPM Tools in a virtual environment. 

Python version
--------------
DPM Tools requires at least Python 3.8, but **Python 3.10+** is recommended.


Required dependencies
--------------
DPM Tools requires the following packages:

- numpy
- matplotlib
- pyvista[all]
- pandas
- tifffile 
- exifread
- netcdf4
- h5py
- porespy
- edt
- scikit-fmm
- connected-components-3d
- pyfftw

Optional dependencies
---------------------
DPM Tools also includes some optional dependencies. For now, these are limited to the optional segmentation module, which include:

+--------------------+-----------------------------------------+
| Package            | Purpose                                 |
+====================+=========================================+
| ``dpm_srm``        | Statistical region merging segmentation |
+--------------------+-----------------------------------------+
| ``dpm_srg``        | Seeded region growing segmentation      |
+--------------------+-----------------------------------------+

Look out for more modules in the future!


Install stable release
----------------------
Stable releases can be installed from `PyPI <https://pypi.org/project/dpm-tools>` using ``pip``::
   pip install dpm_tools

To install with the optional segmentation module, install using ``pip`` with::
   pip install dpm_tools[segment]


Installing the Development Branch from GitHub 
---------------------------------------------
You can install the latest version from GitHub by cloning `DPM Tools <https://github.com/digital-porous-media/dpm_tools>`_, and running::
   git clone https://github.com/digital-porous-media/dpm_tools.git
   cd dpm_tools
   python -m pip install -e .[all]

This will install DPM Tools and development tools for testing and building documentation in an editable environment.