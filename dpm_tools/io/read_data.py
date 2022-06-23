import os
from tifffile import imread as tiffread
import numpy as np


def _read_tiff(filepath: str, full_path: bool = True, **kwargs) -> np.ndarray:
    """
    A utility function to read in TIFF files
    full_path = True if the entire path to the file is provided in filepath (including the file itself)
    If full_path is false, a filename must be provided as a keyword argument.
    """
    if not full_path:
        try:
            filepath = os.path.join(filepath, kwargs['filename'])
        except FileNotFoundError:
            print('Please provide a filename')
    return tiffread(filepath)


def _read_raw(filepath: str, metadata: dict) -> np.ndarray:
    """
    A utility function to read in RAW files
    Must provide image size as nz, ny, nx, number of bits, signed/unsigned and endianness in kwargs
    """
    bits = metadata['bits']
    signed = metadata['signed']
    byte_order = metadata['byte_order']
    nz = metadata['nz']
    ny = metadata['ny']
    nx = metadata['nx']


    if (bits == 8 and signed.lower() == 'unsigned'):
        dt = 'u1'

    elif (bits == 16 and signed.lower() == 'unsigned'):
        dt = 'u2'

    elif (bits == 32 and signed.lower() == 'unsigned'):
        dt = 'u4'

    elif (bits == 8 and signed.lower() == 'signed'):
        dt = 'i1'

    elif (bits == 16 and signed.lower() == 'signed'):
        dt = 'i2'

    elif (bits == 32 and signed.lower() == 'signed'):
        dt = 'i4'

    elif (bits == 32 and signed.lower() == 'real'):
        dt = 'f4'

    elif (bits == 64 and signed.lower() == 'real'):
        dt = 'f8'
    else:
        KeyError("Invalid datatype")

    # Assign byte order based on input
    if (byte_order.lower() == 'little'):
        bt = '<'
    elif (byte_order.lower() == 'big'):
        bt = '>'
    else:
        bt = '|'

    datatype = bt + dt

    # Load the image into an array and reshape it
    return np.fromfile(filepath, dtype=np.uint8).reshape([nz, ny, nx])



def read_image(read_path: str, **kwargs) -> np.ndarray:
    """
    A general use function for reading in an image of any filetype
    Currently supports reading in tiff and raw.
    """
    filetypes = {'tiff': _read_tiff,
                 'tif': _read_tiff,
                 'raw': _read_raw}

    filetype = read_path.rsplit('.', 1)[1]

    try:
        return filetypes[filetype.lower()](read_path, kwargs)
    except NotImplemented:
        print('Cannot read supplied filetype yet')