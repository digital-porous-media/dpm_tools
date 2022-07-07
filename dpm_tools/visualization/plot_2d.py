import matplotlib.pyplot as plt
import numpy as np

from ._vis_utils import _make_dir, _write_hist_csv
from ..__init__ import timer

@timer
def hist(data,
         nbins: int = 256,
         write_csv: bool = False,
         **kwargs):
    """
    Generate a histogram
    If save_fig is True, a save path should be supplied to kwargs under key "filepath"
    """

    # Make line between bars black
    if 'edgecolor' not in kwargs:
        kwargs['edgecolor'] = 'k'

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

    # Create a figures directory if save path is not specified
    if save_path is None:
        save_path = f'{_make_dir("./figures")}'

    # TODO add write_csv with proper savepath
    if write_csv:
        _write_hist_csv(freq, bins, './figures/histogram_csv.csv')

    # TODO add savefig?

    return fig

@timer
def plot_slice(data, slice_z: int = None, slice_axis: int = -1, **kwargs):

    if 'origin' not in kwargs:
        kwargs['origin'] = 'lower'

    if 'interpolation' not in kwargs:
        kwargs['interpolation'] = 'none'

    if slice_z is None:
        slice_z = data.image.shape[slice_axis] // 2

    show_slice = data.image.take(indices=slice_z, axis=slice_axis)


    fig = plt.figure(dpi=400)
    plt.imshow(show_slice, **kwargs)
    plt.axis('off')
    plt.colorbar()
    plt.show()

    return fig

@timer
def thumbnail():
    a = 1

