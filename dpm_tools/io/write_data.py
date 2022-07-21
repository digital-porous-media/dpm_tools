import os
from tifffile import imwrite as tiffwrite
import numpy as np


def _write_tiff(save_path: str, save_filename: str, image_stack: np.ndarray, compression_type: bool = True, tiffSize: bool = False, *args, **kwargs) -> None:
    # TODO implement flags for bigtiff, compression
    
    if compression_type:
        tiffwrite(os.path.join(save_path, save_filename), image_stack,
                bigtiff=tiffSize, compression='zlib', photometric='minisblack')
    else:
        tiffwrite(os.path.join(save_path, save_filename), image_stack,
                  bigtiff=tiffSize, photometric='minisblack')


def _write_raw(save_path: str, save_filename: str, image_stack: np.ndarray, *args, **kwargs) -> None:
    image_stack.tofile(os.path.join(save_path, save_filename))
       

def _write_nc(save_path: str, save_filename: str, image_stack: np.ndarray, *args, **kwargs) -> None:
    image_stack.tofile(os.path.join(save_path, save_filename))


def write_image(save_path: str, save_name: str, image: np.ndarray, filetype: str = 'tiff', **kwargs):
    filetypes = {'tiff': _write_tiff,
                 'tif': _write_tiff,
                 'raw': _write_raw,
                 'nc': _write_nc}

    # TODO Add error catching in filetype
    # try:
    filetypes.get(filetype.lower(), _not_implemented)(save_path, save_name, image, **kwargs)
    # filetypes[filetype.lower()](save_path, save_name, image, **kwargs)

    # except NotImplemented:
    #     print('Save filetype has not been implemented yet')
