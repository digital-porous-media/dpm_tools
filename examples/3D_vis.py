import sys
#sys.path.append('..\..\dpm_tools') #Add custom filepath here if needed
from dpm_tools.io import ImageFromFile, Vector
from dpm_tools.visualization import orthogonal_slices, plot_isosurface, bounding_box, plot_streamlines, plot_scalar_volume
from dpm_tools.visualization import plot_slice, plot_glyph
from dpm_tools.visualization import hist
import numpy as np
import pyvista as pv
import matplotlib.pyplot as plt
from hdf5storage import loadmat
if __name__ == '__main__':
    # image_info = {
    #     'bits': 8,
    #     'signed': 'unsigned',
    #     'byte_order': 'little',
    #     'nx': 365,
    #     'ny': 255,
    #     'nz': 225,
    # }
    #
    # img = ImageFromFile(basepath='../data/', filename='multiphase_ketton.raw', meta=image_info)
    #
    # plot_slice(img)
    #
    # my_fig = plot_orthogonal_slices(img, fig_kwargs={'cmap': 'binary_r'})
    # my_fig.show()
    #
    # my_fig = plot_contours(img, show_isosurface=[2.5], mesh_kwargs={'color': (77, 195, 255), 'opacity': 0.5,
    #                                                                 'smooth_shading': True})
    #
    # my_fig = plot_contours(img, fig= my_fig, show_isosurface=[1.5],
    #                        mesh_kwargs={'color': (0, 255, 0), 'opacity': 0.9})
    #
    # #
    # my_fig.show()
    #file_data = loadmat('C:/Users/bchan/Documents/DPM_Tools/data/10_01_256_elec.mat')
    file_data = loadmat('../data/10_01_256_elec.mat')
    # a, b, c = img['phi'].shape
    binary_img = file_data['phi'] != 0
    print(np.count_nonzero(binary_img))
    img = Vector(image=binary_img, scalar=file_data['phi'], vector=[file_data['Ix'], file_data['Iy'], file_data['Iz']])

    # my_fig = plot_glyph(img, glyph_space=1, mesh_kwargs={'cmap': 'turbo'})
    # my_fig = plot_streamlines(img, tube_radius=0.75)

    my_fig = plot_isosurface(img, show_isosurface=[0.5],
                           mesh_kwargs={'color': (255, 255, 255), 'opacity': 0.15})

    my_fig = plot_scalar_volume(img, fig=my_fig)
    my_fig.show()
