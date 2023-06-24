# -*- coding: utf-8 -*-
"""
Created on Tue Jun 13 23:12:09 2023

@author: cinar
"""

# Importing necessary packages
import os
from tifffile import imread as tiffread
import numpy as np
import pyvista as pv

import skimage.transform

from dpm_tools.io import Image, Vector, ImageFromFile
from dpm_tools.io.io_utils import download_file_url, image_statistics
from dpm_tools.io.read_data import read_image, _read_tiff
from dpm_tools.visualization.plot_2d import hist, plot_slice, make_gif, make_thumbnail
from dpm_tools.visualization._3d_vis_utils import _initialize_plotter, _wrap_array
from dpm_tools.visualization.plot_3d import (plot_orthogonal_slices, plot_contours, 
                                             plot_streamlines, plot_glyph, bounding_box)


sample = _read_tiff('./data/3_fractures.tif')
image_statistics(sample, plot_histogram = True)
img_sample = Image(sample)
plot_slice(img_sample,slice_num=1)

file_url = "https://www.digitalrocksportal.org/projects/125/images/101255/download/"
filename = "Ketton_segmented_oil_blob.raw"

sample_ketton = download_file_url(file_url,filename)

width  = 365
height = 255
slices = 225
  
sample_ketton = np.fromfile(filename, dtype=np.ubyte, sep="")
image = sample_ketton.reshape([slices, height, width])

image_statistics(sample_ketton,plot_histogram=True)
image_rescaled  = skimage.transform.rescale(image.astype(float), 0.5, preserve_range=True)
image_statistics(image_rescaled, plot_histogram=False)

img_ketton = Image(image_rescaled)
plot_slice(img_ketton,slice_num=50)


# Function call to plot the orthogonal slices.
plotter = _initialize_plotter()
plot_orthogonal_slices(img_ketton, fig=plotter)
plotter.show()


# f = plot_streamlines(vector_object, fig=plotter)
plotter = _initialize_plotter()
plot_contours(img_ketton ,fig=plotter)
plotter.show()

bounding_box(img_ketton, fig=plotter)
plotter.show()

image_info = {
    'bits': 8,
    'signed': 'unsigned',
    'byte_order': 'little',
    'nx': 365,
    'ny': 255,
    'nz': 225,
}
new_format = ImageFromFile(basepath='./data/', filename='multiphase_ketton.raw', meta=image_info)
make_gif(new_format)

make_thumbnail(new_format)

# Working with velocity fields
vx = np.load('data/3d_spherepack_project175/vx_rsz_100.npy')
vy = np.load('data/3d_spherepack_project175/vy_rsz_100.npy')
vz = np.load('data/3d_spherepack_project175/vz_rsz_100.npy')

v = np.sqrt(vx**2 + vy**2 + vz**2)

vec = Vector(image=v, scalar=v, vector=[vx,vy,vz])
# compute velocity magnitude

# plot a slice
plot_slice(vec,slice_num=50)

plotter = _initialize_plotter()
plot_orthogonal_slices(vec, fig=plotter)
plotter.show()



plotter = _initialize_plotter()
plot_glyph(vec, fig=plotter, glyph_space=5) # requires pyvista for scale
plotter.show()

plotter = _initialize_plotter()
plot_streamlines(vec, fig=plotter) # we may embed plotter show part inside the function
# plotter.show()

# plotter = _initialize_plotter()
bounding_box(vec, fig=plotter)
plotter.show()
