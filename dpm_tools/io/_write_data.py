import os
from tifffile import imwrite as tiffwrite
import numpy as np
import pathlib


def _write_tiff(save_path: pathlib.Path, save_filename: str, image_stack: np.ndarray, compress: bool = True,
                use_bigtiff: bool = False, **kwargs) -> None:
    """
    Write a 3D image to a tiff file

    Parameters:
        save_path: pathlib.Path of directory to save the file to
        save_filename: str with name of the file
        image_stack: 3D ndarray of the image
        compress: If true: compression using "zlib" by default.
        use_bigtiff: If true, save using bigtiff
        **kwargs: Keyword arguments accepted by ```tifffile.imwrite()``` function

    Returns:
        None

    """
    if not (save_filename.suffix == '.tiff' or save_filename.suffix == '.tif'):
        save_filename = save_filename + '.tiff'

    if compress:
        if "compression" not in kwargs.keys():
            compression = "zlib"
        tiffwrite(os.path.join(save_path, save_filename), image_stack,
                  bigtiff=use_bigtiff, compression=compression, photometric='minisblack', **kwargs)
    else:
        tiffwrite(os.path.join(save_path, save_filename), image_stack,
                  bigtiff=use_bigtiff, photometric='minisblack', **kwargs)


def _write_raw(save_path: pathlib.Path, save_filename: str, image_stack: np.ndarray, **kwargs) -> None:
    """
    Write a 3D image to a raw file

    Parameters:
        save_path: pathlib.Path of directory to save the file to
        save_filename: Save name of the file
        image_stack: 3D ndarray of the image
        **kwargs: Keyword arguments for ```numpy.tofile()``` function

    Returns:
        None
    """
    image_stack.tofile(os.path.join(save_path, save_filename))
       
# TODO implement netcdf
# def _write_nc(save_path: str, save_filename: str, image_stack: np.ndarray, **kwargs) -> None:
#     """
#     Write a 3D image to a netcdf file
#
#     Parameters:
#         save_path: pathlib.Path of directory to save the file to
#         save_filename: Save name of the file
#         image_stack: 3D ndarray of the image
#         **kwargs: Keyword arguments for ```
#
#     """
#
#     image_stack.tofile(os.path.join(save_path, save_filename))


def write_image(save_path: pathlib.Path, save_name: str, image: np.ndarray, filetype=None, **kwargs) -> None:
    """
    Write a 3D image in a specified filetype

    Parameters:
        save_path: pathlib.Path of the directory to save
        save_name: name of the file to save. If filetype is not specified, must contain the extension
        image: 3D ndarray image to save
        filetype: filetype extension to save. If specified, overrides the save_name extension.

    Returns:
        None
    """

    filetypes = {'.tiff': _write_tiff,
                 '.tif': _write_tiff,
                 '.raw': _write_raw,}
                 # '.nc': _write_nc}

    if isinstance(save_path, str):
        save_path = pathlib.Path(save_path)

    if filetype is None:
        filetype = pathlib.Path(save_name).suffix

    filetype = filetype.lower()

    assert filetype in filetypes, f"Invalid filetype {filetype}"
    if not filetype.startswith('.'):
        filetype = "." + filetype

    save_name = pathlib.Path(save_name).with_suffix(filetype)

    # TODO Add error catching in filetype
    filetypes[filetype.lower()](save_path, save_name, image, **kwargs)
