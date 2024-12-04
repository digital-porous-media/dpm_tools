import os
from tifffile import imread as tiffread
import numpy as np
import string
import netCDF4 as nc
import h5py
from scipy.io import loadmat
from dataclasses import dataclass, field
import pathlib
from collections.abc import Iterable

# __all__ = [
#     'read_image',
#     'ImageFromFile',
#     'Image',
#     'Vector'
# ]


def _read_tiff(filepath: pathlib.Path, full_path: bool = True, **kwargs) -> np.ndarray:
    """
    A utility function to read in TIFF files

    Parameters:
        filepath: pathlib.Path to the tiff file
        full_path: If false, a filename must be provided as a keyword argument.

    Returns:
        np.ndarray: Read in image
    """
    if not full_path:
        try:
            filepath = filepath / kwargs['filename']
        except FileNotFoundError:
            print('Please provide a filename')

    return tiffread(filepath)


def _read_raw(filepath: pathlib.Path, **kwargs) -> np.ndarray:
    """
    A utility function to read in RAW files


    Parameters:
        filepath: pathlib.Path to the raw file
        **kwargs: Must provide a ```meta``` dictionary as a kwarg with keys nz, ny, nx, bits, signed and byte_order

    Returns:
        np.ndarray: Read in image
    """

    assert 'meta' in kwargs, "Image metadata dictionary is required to contain " \
                             "keywords 'nz', 'ny', 'nx', 'bits','signed' and 'byte_order'"

    metadata = kwargs['meta']

    assert all(key in metadata for key in ['nz', 'ny', 'nx', 'bits', 'signed', 'byte_order']), \
        f"Image metadata dictionary must contain {
            ['nz', 'ny', 'nx', 'bits', 'signed', 'byte_order']} - found {list(metadata.keys())}"

    bits = metadata['bits']
    signed = metadata['signed']
    byte_order = metadata['byte_order']
    nz = metadata['nz']
    ny = metadata['ny']
    nx = metadata['nx']

    # Assign data type based on input
    if (signed.lower() == 'unsigned'):
        dt1 = 'u'

    elif (signed.lower() == 'signed'):
        dt1 = 'i'

    elif (signed.lower() == 'real'):
        dt1 = 'f'

    dt2 = bits // 8

    # Assign byte order based on input
    if (byte_order.lower()[:3] == 'lit'):
        bt = '<'
    elif (byte_order.lower()[:3] == 'big'):
        bt = '>'
    else:
        bt = '|'

    datatype = bt + dt1 + str(dt2)

    # Load the image into an array and reshape it
    return np.fromfile(filepath, dtype=datatype).reshape([nz, ny, nx])

# TODO: Review this function


def _read_nc(filepath: pathlib.Path, **kwargs) -> np.ndarray:
    """
    A utility function to read in netcdf files

    Parameters:
        filepath: pathlib.Path of the path to the netcdf file

    Returns:
        np.ndarray: Read in image
    """

    ds = nc.Dataset(filepath)

    array_name = ""

    # Search metadata variables for the image array name
    for var in ds.variables.values():
        str_var = str(var)
        var_parts = str_var.split("\n")

        # Find the shape variable
        for part in var_parts:
            if "current shape" in part:
                find_numbers = part.split(" ")
                dimension_count = 0

                # Search for arrays with more than 1 dimension
                for num_part in find_numbers:
                    if (num_part.isalpha() == False and num_part not in string.punctuation):
                        dimension_count += 1

        # Extract the name from the correct array
        if dimension_count > 1:
            var_list1 = var_parts[1]
            find_name = var_list1.split(" ")
            i = 0

            # Add the name to a string
            while (find_name[1][i] not in string.punctuation):
                array_name += find_name[1][i]
                i += 1

    # Load the image into an array

    image_array = ds[array_name][:]

    return image_array


def _read_mat(filepath: pathlib.Path, data_keys: str = None, **kwargs) -> np.ndarray:
    """
    A utility function to read in .mat (matlab) files

    Parameters:
        filepath: pathlib.Path of the path to the .mat file
        data_keys: a string with the key to be used for reading in the data

    Returns:
        np.ndarray: The image array
    """
    try:
        with h5py.File(filepath, 'r') as data:
            if data_keys is None:
                image = {k: data[k][:] for k in [*data.keys()]}
                # image = data
                # image = [data[k][:] for k in [*data.keys()]]

            else:
                try:
                    image = data[data_keys][:]
                except KeyError:
                    print("Key could not be found in this file")
                    image = []
    except OSError:
        data = loadmat(filepath)
        if data_keys is None:
            image = {k: data[k] for k in [*data.keys()]}
            # image = data
            # image = [data[k] for k in [*data.keys()]]

        else:
            try:
                image = data[data_keys]
            except KeyError:
                print("Key could not be found in this file")
                image = []

    return image


def _not_implemented():
    raise NotImplementedError("No support for this datafile type... yet")


def read_image(read_path: str, **kwargs) -> np.ndarray:
    """
    A general use function for reading in an image of the implemented filetypes
    Currently supports reading in tiff, raw, and mat.

    Parameters:
        read_path: pathlib.Path of the file to read in
        **kwargs: Required keyword arguments for the utility functions

    Returns:
        np.ndarray: The image array
    """
    read_path = pathlib.Path(read_path)

    filetypes = {'.tiff': _read_tiff,
                 '.tif': _read_tiff,
                 '.raw': _read_raw,
                 '.nc': _read_nc,
                 '.mat': _read_mat}

    filetype = read_path.suffix

    return filetypes.get(filetype.lower(), _not_implemented)(read_path, **kwargs)


@dataclass(kw_only=True)
class Image:
    """
    Image dataclass. At least one of scalar, vector, or filepath attributes must be provided.

    Attributes:
        scalar (np.ndarray): A 3D array representing scalar values of an image (e.g. binary image, pressure field, etc.)
        vector (list[np.ndarray, np.ndarray, np.ndarray]): A list containing 3 np.ndarrays of the vector components (z, x, y)
        filepaths (list): A list containing the filepaths to the image data. Assumes the order of paths are scalar,
        z-, y-, x- components of the vector field. If only providing vector field data, assign None to the 0th index
        meta (list): A list containing dictionaries for the metadata of each image in filepaths used to read the image.
        If no metadata is necessary, pass a list of empty dictionaries
        shape (tuple): The shape of the 3D ndarray containing the image
        nx (int): Width of the image (in voxels)
        ny (int): Height of the image (in voxels)
        nz (int): Number of slices of the image
        magnitude: Magnitude of vector (if vector is provided)
    """
    scalar: np.ndarray = None
    vector: list = None
    filepaths: list = None
    meta: list = None

    def __post_init__(self):

        if self._check_dataclass_inputs():
            self._read_data_from_files()

        assert self.scalar is not None or self.vector is not None, "Provide either scalar or vector data"

        # Check if only scalar
        if self.scalar is not None and self.vector is None:
            if self.scalar.ndim == 2:
                self.scalar = self.scalar[np.newaxis, :, :]
            self.nz, self.nx, self.ny = self.scalar.shape

        # Check if only vector
        if self.scalar is None and self.vector is not None:
            assert self.vector[0].shape == self.vector[1].shape == self.vector[2].shape, "Vector shapes must be consistent"
            # Assign scalar to be where vector value == 0
            self.scalar = (self.vector[0] == 0).astype(np.uint8)

        # If scalar and vector are both populated
        if self.scalar is not None and self.vector is not None:
            assert self.scalar.shape == self.vector[0].shape == self.vector[1].shape == self.vector[2].shape, \
                "Vector shapes must be consistent with scalar shape"

        # If vector is populated
        if self.vector is not None:
            self.magnitude = np.sqrt(
                self.vector[0]**2 + self.vector[1]**2 + self.vector[2]**2)

        self.nz, self.nx, self.ny = self.scalar.shape
        self.shape = (self.nz, self.nx, self.ny)

    def _check_dataclass_inputs(self) -> bool:
        """
        A utility method to check what was inputed.
        """
        read_from_file = False
        if self.filepaths is None:
            assert self.scalar is not None or self.vector is not None
        else:
            assert len(self.meta) == len(self.filepaths), \
                f"Provide a list of metadata for each image provided in the filepaths"
            read_from_file = True

        return read_from_file

    def _read_data_from_files(self):
        """
        A utility method to read images in from a list of filepaths
        """

        images = [read_image(fp, meta=self.meta[i])
                  if fp else None for i, fp in enumerate(self.filepaths)]

        assert len(images) == 1 or len(images) == 3 or len(
            images) == 4, f"You must provide a scalar, a vector, or both"

        if len(images) == 1:
            self.scalar = images[0]
        elif len(images) == 3:
            self.vector = images
        elif len(images) == 4:
            self.scalar = images[0]
            self.vector = images[1:]
