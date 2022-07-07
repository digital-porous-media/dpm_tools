import matplotlib.pyplot as plt
import matplotlib.animation as anim
import numpy as np
from tqdm import tqdm

from ._vis_utils import _make_dir, _write_hist_csv, _scale_image, AnimatedGif
from ..__init__ import timer


# TODO Add fig save decorator

@timer
def hist(data,
         nbins: int = 256,
         write_csv: bool = False,
         **kwargs):
    """
    Generate a histogram
    If save_fig is True, a save path should be supplied to kwargs under key "filepath"
    """

    # # Make line between bars black
    # if 'edgecolor' not in kwargs:
    #     kwargs['edgecolor'] = 'k'

    # Set default figure size
    if 'fig_size' not in kwargs:
        kwargs['fig_size'] = (4, 2.4)

    # Make the histogram
    fig = plt.figure(figsize=kwargs['fig_size'])
    kwargs.pop('fig_size', None)  # Remove fig_size argument from kwargs
    freq, bins, _ = plt.hist(x=data.image.ravel(), bins=nbins, density=True, **kwargs)
    plt.xlabel('Gray value')
    plt.ylabel('Probability')
    plt.tight_layout()
    plt.show()

    # # Create a figures directory if save path is not specified
    # if save_path is None:
    #     save_path = f'{_make_dir("./figures")}'

    # TODO add write_csv with proper savepath
    if write_csv:
        _write_hist_csv(freq, bins, './figures/histogram_csv.csv')

    # TODO add savefig?

    return fig

@timer
def plot_slice(data, slice_num: int = None, slice_axis: int = 0, **kwargs):

    if 'origin' not in kwargs:
        kwargs['origin'] = 'lower'

    if 'interpolation' not in kwargs:
        kwargs['interpolation'] = 'none'

    if slice_num is None:
        slice_num = data.image.shape[slice_axis] // 2

    show_slice = data.image.take(indices=slice_num, axis=slice_axis)


    fig = plt.figure(dpi=400)
    plt.imshow(show_slice, **kwargs)
    plt.axis('off')
    plt.colorbar()
    plt.show()

    return fig

@timer
def make_thumbnail(data, thumb_slice: int = None, fig_size: tuple = (1, 1), slice_axis: int = 0, **kwargs):
    if thumb_slice is None:
        thumb_slice = int(np.floor(data.image.shape[slice_axis] / 2))

    fig = plt.figure()
    fig.set_size_inches(fig_size)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    plt.set_cmap('Greys')
    show_slice = _scale_image(data.image).take(indices=thumb_slice, axis=slice_axis)
    ax.imshow(show_slice, aspect='equal', vmin=0, vmax=1, **kwargs)
    plt.show()

    return fig


@timer
def make_gif(data,  dpi: int = 96, **kwargs):
    """
    Function to make and save a gif
    """

    sl1 = data.image[0, :, :]
    images = []
    animated_gif = AnimatedGif()
    animated_gif.add(sl1, h=data.nx, w=data.ny)
    slices = data.nz
    print("Animation created for slice 1/" + str(slices))

    if slices < 20:
        slicesave = 1
    elif slices < 100:
        slicesave = 5
    elif slices < 250:
        slicesave = 12
    else:
        slicesave = 20

    for i in range(1, slices, slicesave):
        sl = data.image[i, :, :]
        animated_gif.add(sl, h=data.nx, w=data.ny)
        print("Animation created for slice " + str(i + 1) + "/" + str(slices))


    animated_gif.save(f'{data.basepath}/{data.basename}.gif')

    #
    # # Save every ~n_slices/20 slice in gif
    # if data.nz <= 20:
    #     slice_save = 1
    # else:
    #     slice_save = int(np.round(data.nz / 20))
    #
    # images = [None]*20
    #
    # fig = plt.figure()
    #
    #
    # for i, j in enumerate(range(0, data.nz, slice_save)):
    #     fig.set_size_inches(data.nx / dpi, data.ny / dpi)
    #     ax1 = plt.Axes(fig, [0., 0., 1., 1.])
    #     ax1.set_axis_off()
    #     fig.add_axes(ax1)
    #     plt.set_cmap('Greys')
    #     image_slice = _scale_image(data.image.take(indices=j, axis=0))
    #     images[i] = ax1.imshow(image_slice, vmin=0, vmax=255, **kwargs)
    #     plt.show()

    # animation = anim.ArtistAnimation(fig, images)
    # animation.save(f"{data.basepath}/{data.basename}.gif", writer='imagemagick', fps=60)

    return images


