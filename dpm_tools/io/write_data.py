import os
from tifffile import imwrite as tiffwrite
import numpy as np


def _write_tiff(save_path: str, save_filename: str, image_stack: np.ndarray) -> None:
    # TODO implement flags for bigtiff, compression
    tiffwrite(os.path.join(save_path, save_filename), image_stack,
              bigtiff=True, compression='zlib', photometric='minisblack')


def _write_raw(save_path: str, save_filename: str, image_stack: np.ndarray) -> None:
    image_stack.tofile(os.path.join(save_path, save_filename))


def write_image(save_path: str, save_name: str, image: np.ndarray, filetype: str = 'tiff', **kwargs):
    filetypes = {'tiff': _write_tiff,
                 'tif': _write_tiff,
                 'raw': _write_raw}

    try:
        filetypes[filetype.lower()](save_path, save_name, image, kwargs)

    except NotImplemented:
        print('Save filetype has not been implemented yet')
