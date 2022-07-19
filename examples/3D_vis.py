from dpm_tools.io import Image
from dpm_tools.visualization import plot_orthogonal_slices, plot_contours, bounding_box
from dpm_tools.visualization import plot_slice

if __name__ == '__main__':
    image_info = {
        'bits': 8,
        'signed': 'unsigned',
        'byte_order': 'little',
        'nx': 365,
        'ny': 255,
        'nz': 225,
    }

    img = Image(basepath='../data/', filename='multiphase_ketton.raw', meta=image_info)

    plot_slice(img)

    my_fig = plot_orthogonal_slices(img, fig_kwargs={'cmap': 'binary_r'})
    my_fig.show()

    my_fig = plot_contours(img, show_isosurface=[2.5], mesh_kwargs={'color': (77, 195, 255), 'opacity': 0.5,
                                                                    'smooth_shading': True})

    my_fig = plot_contours(img, fig= my_fig, show_isosurface=[1.5],
                           mesh_kwargs={'color': (0, 255, 0), 'opacity': 0.9})

    #
    my_fig.show()
