# -*- coding: utf-8 -*-
"""
Created on Tue Jun 13 23:12:09 2023

@author: cinar
"""

# Importing necessary packages
import numpy as np

from dpm_tools.io import Image, Vector, ImageFromFile, read_image
from dpm_tools.visualization._plot_2d import hist, plot_slice, make_gif
from dpm_tools.visualization._plot_3d import orthogonal_slices, plot_isosurface, plot_streamlines
from dpm_tools.visualization._plot_3d import plot_glyph, bounding_box

# *************************************************************************** #
# Fundamentals ************************************************************** #
# *************************************************************************** #

## Reading an image by passing the directory to the function read_image:
image = read_image('../data/35_1.tiff')

## Converting the image to "Image" class
img_sample = Image(image)

## Observing the histogram of the gray values and the plot of the image
hist(img_sample)
plot_slice(img_sample, cmap='gray')


## Reading a new image, after defining the metadata.
metadata = {
    'bits': 8,
    'signed': 'unsigned',
    'byte_order': 'little',
    'nx': 365,
    'ny': 255,
    'nz': 225,
}
  
sample_ketton = read_image("./Ketton_segmented_oil_blob.raw", meta=metadata)
img_ketton = Image(sample_ketton)

## Since this image has 225 slices, the slice number can be indicated in the function.
plot_slice(img_ketton,slice_num=112)

## Function call to plot the orthogonal slices.
fig_orthogonal = orthogonal_slices(img_ketton,slider=True)
fig_orthogonal.show()


## Function call to plot the contours. If the 'show_isosurfaces' is not given,
## the middle value of the isosurface range will be used as default.

fig_contours = plot_isosurface(img_ketton,show_isosurface=[1.5,2.5]) 
fig_contours.show()

# *************************************************************************** #
# Creating a 'gif' from an image, using ImageFromFile function ************** #
# *************************************************************************** #

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

# *************************************************************************** #
# Working with velocity fields ********************************************** #
# *************************************************************************** #

## Importing the velocity data
vx = np.load('../data/3d_spherepack_project175/vx_rsz_100.npy')
vy = np.load('../data/3d_spherepack_project175/vy_rsz_100.npy')
vz = np.load('../data/3d_spherepack_project175/vz_rsz_100.npy')

## Getting the magnitude
v = np.sqrt(vx**2 + vy**2 + vz**2)

## Creating a Vector class, specifying the image, the scalar data, and the vectors
## Here, the image and the scalar is composed of the magnitude since there is only
## velocity vectors are available
vec = Vector(image=v, scalar=v, vector=[vx,vy,vz])


## Plotting a slice
plot_slice(vec,slice_num=50) 


fig_orthogonal_velocity = orthogonal_slices(vec,slider=True)
fig_orthogonal_velocity.show()


## Plotting glyph
fig_glyph = plot_glyph(vec)
plot_isosurface(vec, fig=fig_glyph)
# bounding_box(vec, fig=fig_glyph)
fig_glyph.show()

## Plotting the streamlines with a bounding box
fig_streamlines = plot_streamlines(vec)
plot_isosurface(vec, fig=fig_streamlines, show_isosurface=3)

## Adding a bounding box to the streamlines
bounding_box(vec, fig=fig_streamlines)
fig_streamlines.show()


# *************************************************************************** #
# Working with velocity fields - Example 2 ********************************** #
# *************************************************************************** #

## Importing the image:
metadata = {
    'bits': 8,
    'signed': 'unsigned',
    'byte_order': 'little',
    'nx': 1000,
    'ny': 1000,
    'nz': 1000,
}

bentheimer_ss_scalar = read_image("data/Bentheimer Sandstone/bentheimer.raw", meta=metadata)

# Reducing the size for faster visualization
bentheimer_ss_scalar = bentheimer_ss_scalar[105 : 205,155 : 255,115 : 215]
bentheimer_ss = Image(bentheimer_ss_scalar)

bentheimer_ss_vx = np.fromfile("data/Bentheimer Sandstone/Ux.raw", dtype=np.float64)
bentheimer_ss_vx = bentheimer_ss_vx.reshape((500,500,500))

bentheimer_ss_vy = np.fromfile("data/Bentheimer Sandstone/Uy.raw", dtype=np.float64)
bentheimer_ss_vy = bentheimer_ss_vy.reshape((500,500,500))

bentheimer_ss_vz = np.fromfile("data/Bentheimer Sandstone/Uz.raw", dtype=np.float64)
bentheimer_ss_vz = bentheimer_ss_vz.reshape((500,500,500))

bentheimer_ss_vx = bentheimer_ss_vx[105 : 205,155 : 255,115 : 215]
bentheimer_ss_vy = bentheimer_ss_vy[105 : 205,155 : 255,115 : 215]
bentheimer_ss_vz = bentheimer_ss_vz[105 : 205,155 : 255,115 : 215]


## Creating a Vector class, specifying the image, the scalar data, and the vectors
## Here, the image and the scalar is composed of the magnitude since there is only
## velocity vectors are available
bentheimer_ss_vector = Vector(image=bentheimer_ss_scalar, scalar=bentheimer_ss_scalar, 
                              vector=[bentheimer_ss_vx,
                                      bentheimer_ss_vy,
                                      bentheimer_ss_vz])


## Plotting a slice
plot_slice(bentheimer_ss_vector,slice_num=50)

fig_orthogonal_velocity = orthogonal_slices(bentheimer_ss_vector,slider=True)
fig_orthogonal_velocity.show()


## Plotting glyph
fig_glyph = plot_glyph(bentheimer_ss_vector, glyph_space=1) # Scaled the arrows
plot_isosurface(bentheimer_ss_vector, fig=fig_glyph,
                           mesh_kwargs={'color': (255, 255, 255), 'opacity': 0.15})
fig_glyph.show()

## Plotting the streamlines with a bounding box
fig_streamlines = plot_streamlines(bentheimer_ss_vector)

## Adding a bounding box to the streamlines
bounding_box(bentheimer_ss_vector, fig=fig_streamlines)
plot_isosurface(bentheimer_ss_vector, fig=fig_streamlines,
                           mesh_kwargs={'color': (255, 255, 255), 'opacity': 0.15})
fig_streamlines.show()

# *************************************************************************** #
# Working with velocity fields - Example 3 ********************************** #
# *************************************************************************** #

## Importing the image:
metadata = {
    'bits': 8,
    'signed': 'unsigned',
    'byte_order': 'little',
    'nx': 500,
    'ny': 500,
    'nz': 500,
}

estaillades_carbonate_scalar = read_image("data/Estaillades Carbonate/estaillades.raw", meta=metadata)

# Reducing the size for faster visualization
estaillades_carbonate_scalar = estaillades_carbonate_scalar[235:335,235:335,235:335]
estaillades_carbonate = Image(estaillades_carbonate_scalar)

estaillades_carbonate_vx = np.fromfile("data/Estaillades Carbonate/Ux.raw", dtype=np.float64)
estaillades_carbonate_vx = estaillades_carbonate_vx.reshape((500,500,500))

estaillades_carbonate_vy = np.fromfile("data/Estaillades Carbonate/Uy.raw", dtype=np.float64)
estaillades_carbonate_vy = estaillades_carbonate_vy.reshape((500,500,500))

estaillades_carbonate_vz = np.fromfile("data/Estaillades Carbonate/Uz.raw", dtype=np.float64)
estaillades_carbonate_vz = estaillades_carbonate_vz.reshape((500,500,500))

estaillades_carbonate_vx = estaillades_carbonate_vx[253 : 353,90 : 190,318 : 418]
estaillades_carbonate_vy = estaillades_carbonate_vy[253 : 353,90 : 190,318 : 418]
estaillades_carbonate_vz = estaillades_carbonate_vz[253 : 353,90 : 190,318 : 418]


## Creating a Vector class, specifying the image, the scalar data, and the vectors
## Here, the image and the scalar is composed of the magnitude since there is only
## velocity vectors are available

binary_image = estaillades_carbonate_scalar != 0
estaillades_carbonate_vector = Vector(image=estaillades_carbonate_scalar, 
                                      scalar=estaillades_carbonate_scalar, 
                              vector=[estaillades_carbonate_vx,
                                      estaillades_carbonate_vy,
                                      estaillades_carbonate_vz])


## Plotting a slice
plot_slice(estaillades_carbonate_vector,slice_num=50)


fig_orthogonal_velocity = orthogonal_slices(estaillades_carbonate_vector, slider=True)
fig_orthogonal_velocity.show()


## Plotting glyph
fig_glyph = plot_glyph(estaillades_carbonate_vector, glyph_space=1) # Scaled the arrows
bounding_box(estaillades_carbonate_vector, fig=fig_glyph)
plot_isosurface(estaillades_carbonate_vector, fig=fig_glyph,
                           mesh_kwargs={'color': (255, 255, 255), 'opacity': 0.15})
fig_glyph.show()

## Plotting the streamlines with a bounding box
fig_streamlines = plot_streamlines(estaillades_carbonate_vector)

## Adding a bounding box to the streamlines
bounding_box(estaillades_carbonate_vector, fig=fig_streamlines)
plot_isosurface(estaillades_carbonate_vector, fig=fig_streamlines,
                           mesh_kwargs={'color': (255, 255, 255), 'opacity': 0.15})
fig_streamlines.show()

