import os
import sys
import codecs
from setuptools import setup, find_packages
# from Cython.Build import cythonize
import numpy as np

sys.path.append(os.getcwd())
version_path = 'dpm_tools/__version__.py'

this_directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_directory, "README.md"), encoding='utf-8') as f:
    long_description = f.read()

def read(rel_path):
    here = os.path.abspath(os.path.dirname(__file__))
    with codecs.open(os.path.join(here, rel_path), 'r') as fp:
        return fp.read()
def get_version(rel_path):
    for line in read(rel_path).splitlines():
        if line.startswith('__version__'):
            delim = '"' if '"' in line else "'"
            ver = line.split(delim)[1].split(".")
            return ".".join(ver)
    else:
        raise RuntimeError("Unable to find version string.")

setup(
    name='dpm_tools',
    description='A toolkit to process and analyze digital porous media images',
    long_description=long_description,
    long_description_content_type='text/markdown',
    version=get_version(version_path),
    zip_safe=False,
    classifiers=[
        'Development Status :: 1 - Planning',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3 :: Only',
        'Programming Language :: Python :: 3.10'
    ],
    packages=[
        'dpm_tools',
        'dpm_tools.io',
        'dpm_tools.metrics',
        'dpm_tools.visualization'
    ],
    setup_requires=['numpy']
    install_requires=[
        'numpy',
        'matplotlib',
        'pyvista[all]',
        'pandas',
        'tifffile',
        'exifread',
        'netcdf4',
        'hdf5storage',
        'porespy',
        'edt',
        'scikit-fmm',
        'connected-components-3d',
        # 'Cython',
        'pyarrow'
    ],
    # ext_modules=cythonize("dpm_tools/metrics/binary_configs.pyx"),
    # include_dirs=[np.get_include()],
    author='Digital Porous Media Team',
    author_email='bcchang@utexas.edu',
    download_url='https://github.com/digital-porous-media/dpm_tools',
    url='digital-porous-media.github.io/dpm_tools/html',
)
