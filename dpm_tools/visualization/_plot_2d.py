import matplotlib.pyplot as plt
import matplotlib.animation as anim
import matplotlib as mpl
import numpy as np
from itertools import repeat
from tqdm import tqdm
from typing import Any, Tuple

from ._vis_utils import _make_dir, _write_hist_csv, _scale_image
from ..metrics._feature_utils import _sigmoid


# TODO Add fig save decorator

def hist(data,
         data2: np.array = None,
         nbins: int = 256,
         write_csv: bool = False,
         **kwargs):
    """
    Generate a histogram

    Parameters:
    ___
    :data: The data to plot histogram for.
    :data2: The data to plot histogram for.
    :nbins: The number of bins for the histogram.
    :write_csv: True = write the histogram to a csv file.

    Returns:
        plt.figure: The figure that was generated

    If save_fig is True, a save path should be supplied to kwargs under key "filepath"
    Use data2 for adding a second distribution and plotting them together
    """

    # # Make line between bars black
    if 'edgecolor' not in kwargs:
        kwargs['edgecolor'] = 'k'

    # Set default figure size
    if 'fig_size' not in kwargs:
        kwargs['fig_size'] = (4, 2.4)

    # Make the histogram
    fig = plt.figure(figsize=kwargs['fig_size'])
    kwargs.pop('fig_size', None)  # Remove fig_size argument from kwargs

    if data2 is not None:
        plt.hist(x=data.scalar.ravel(), bins=nbins,
                 density=True, **kwargs, label='data1')
        plt.hist(x=data2.scalar.ravel(), bins=nbins,
                 density=True, **kwargs, label='data2')
        plt.legend()
        plt.xlabel('Gray value')
        plt.ylabel('Probability')
        plt.tight_layout()
        plt.show()
    else:
        freq, bins, _ = plt.hist(
            x=data.scalar.ravel(), bins=nbins, density=True, **kwargs)
        plt.xlabel('Gray value')
        plt.ylabel('Probability')
        plt.tight_layout()
        plt.show()

    # # Create a figures directory if save path is not specified
    # if save_path is None:
    #     save_path = f'{_make_dir("./figures")}'

    # TODO add write_csv with proper savepath
    # if write_csv:
    #     _write_hist_csv(freq, bins, './figures/histogram_csv.csv')

    # TODO add savefig?

    return fig


def plot_slice(data, slice_num: int = None, slice_axis: int = 0, **kwargs):
    if 'origin' not in kwargs:
        kwargs['origin'] = 'lower'

    if 'interpolation' not in kwargs:
        kwargs['interpolation'] = 'none'

    if 'cmap' not in kwargs:
        kwargs['cmap'] = 'viridis'

    if slice_num is None:
        slice_num = data.scalar.shape[slice_axis] // 2

    show_slice = data.scalar.take(indices=slice_num, axis=slice_axis)

    fig = plt.figure(dpi=400)
    plt.imshow(show_slice, **kwargs)
    plt.axis('off')
    plt.colorbar()
    plt.show()

    return fig


def make_thumbnail(data, thumb_slice: int = None, fig_size: tuple = (1, 1), slice_axis: int = 0, **kwargs):
    if thumb_slice is None:
        thumb_slice = int(np.floor(data.scalar.shape[slice_axis] / 2))

    fig = plt.figure()
    fig.set_size_inches(fig_size)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    plt.set_cmap('Greys')
    show_slice = _scale_image(data.scalar).take(
        indices=thumb_slice, axis=slice_axis)
    ax.imshow(show_slice, aspect='equal', vmin=0, vmax=1, **kwargs)
    plt.show()

    return fig


def make_gif(data, dpi: int = 96, save: bool = False, **kwargs):
    """
    Function to make and save a gif
    """
    # Save every ~n_slices/20 slice in gif
    if data.nz <= 20:
        slice_save = 1
    else:
        slice_save = int(np.round(data.nz / 50))
    print(slice_save)
    images = list(repeat([], data.nz // slice_save + 1))

    fig = plt.figure()
    fig.set_size_inches(data.nx / dpi, data.ny / dpi)
    ax1 = plt.Axes(fig, [0., 0., 1., 1.])
    ax1.set_axis_off()
    fig.add_axes(ax1)
    plt.set_cmap('Greys')

    gif_slices = _scale_image(data.scalar[::slice_save])
    images = [[ax1.imshow(slices, vmin=0, vmax=255, **kwargs)]
              for slices in tqdm(gif_slices)]

    animation = anim.ArtistAnimation(fig, images)
    if save:
        animation.save(f"{data.basepath}/{data.basename}.gif",
                       writer='imagemagick', fps=7)
    else:
        plt.show()

    return images


def plot_heterogeneity_curve(radii: np.ndarray, variances: np.ndarray, relative_radii: bool = True, fig=None, ax=None, **kwargs) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot the results of the porosity variance heterogeneity analysis with colored heterogeneous/homogenous zones.

    Parameters:
        radii: The window sizes used to calculate the porosity variance.
        variances: The porosity variances for each window size.
        relative_radii: If True, the plotted radii are relative to the first window size. Otherwise, the absolute radii are shown.

    Returns:
        fig, ax: Matplotlib figure and axes object.
    """
    if fig is None and ax is None:
        fig1, ax1 = plt.subplots()
    else:
        assert fig is not None and ax is not None, "Both fig and ax must be provided."
        fig1, ax1 = fig, ax
    # plt.figure()
    if relative_radii:
        ax1.plot(variances, markersize=6, **kwargs)
        ax1.set_xlabel("Relative Radius")
    else:
        ax1.plot(radii, variances, markersize=6,
                 **kwargs)
        ax1.set_xlabel("Absolute Radius")

    x = np.linspace(-2, 17, len(variances))
    x2 = np.linspace(-2, 6, len(variances))

    bound = (0.023 * (1 - _sigmoid(x)))
    bnd = bound[bound <= 0.0025]
    bound[bound <= 0.0025] = np.linspace(0.0025, 0.001, len(bnd))

    ax1.fill_between(range(len(variances)), bound, facecolor='g',
                     alpha=0.3, label='Homogeneity Zone' if fig is None else None)

    ax1.fill_between(range(len(variances)), bound, ((0.035 * (1 - _sigmoid(x2)))) + 0.007, facecolor='r', alpha=0.3,
                     label='Heterogeneity Zone' if fig is None else None)

    ax1.set_ylabel("Porosity Variance")

    # plt.legend()

    return fig1, ax1
