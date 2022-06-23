import os
from tifffile import imread as tiffread
import numpy as np


def _read_tiff(filepath: str, full_path: bool = True, **kwargs) -> np.ndarray:
    if not full_path:
        try:
            filepath = os.path.join(filepath, kwargs['filename'])
        except FileNotFoundError:
            print('Please provide a filename')
    return tiffread(filepath)


def read_image(read_path: str, **kwargs):
    filetypes = {'tiff': _read_tiff,
                 'tif': _read_tiff}

    filetype = read_path.rsplit('.', 1)[1]

    try:
        filetypes[filetype.lower()](read_path, kwargs)
    except NotImplemented:
        print('Cannot read supplied filetype yet')
