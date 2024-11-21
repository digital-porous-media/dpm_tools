import glob
import os
import tifffile
import numpy as np
import pathlib
from typing import Any, Tuple
import re

from .read_data import read_image
from .write_data import write_image


def find_files_with_ext(directory: pathlib.Path, extension: str) -> list:
    """
    Search the given parent directory for files with the given extension.

    Parameters
        directory: pathlib.Path of the parent directory
        extension: extension of the files to find

    Returns
        list: A list with the file names containing the given extension
    """

    if not isinstance(directory, pathlib.Path):
        directory = pathlib.Path(directory)

    if not extension.startswith("."):
        extension = f".{extension}"

    assert directory.exists(), f"{directory} could not be found"
    assert directory.is_dir(), f"{directory} is not a directory"

    files_found = list(directory.glob(f'**/*{extension}'))

    return files_found


def _get_file_sizes(file_list: list[pathlib.Path]) -> list[int]:
    """
    Get the sizes of all files in the given list in bytes

    Parameters
        file_list: A list of pathlib.Path objects

    Returns
        list: A list of the file sizes in bytes
    """
    if not isinstance(file_list, list):
        file_list = [file_list]

    assert len(file_list) > 0, "File list cannot be empty"

    return [os.path.getsize(object) for object in file_list]


def _get_tiff_tag(tiff_page: tifffile.TiffPage, tag_name: str) -> Any:
    """
    Utility function to get the value of the given tag from the given tiff page

    Parameters
        tiff_page: tifffile.TiffPage object
        tag_name: Name of the tag to get the value of

    Returns
        Any: The value of the given tag
    """
    tag = tiff_page.tags.get(tag_name)
    return tag.value if tag else None


def get_tiff_metadata(directory: pathlib.Path) -> Tuple:
    """
    Get the metadata of all tiff files in the given directory.

    Parameters
        directory: pathlib.Path of the parent directory

    Returns
        pd.DataFrame: DataFrame with the file name, number of slices,
        width, height
    """

    # Find all .tiff files
    tiff_files = find_files_with_ext(directory, ".tif*")
    sizes = _get_file_sizes(tiff_files)
    slices = [None] * len(tiff_files)
    width = [None] * len(tiff_files)
    length = [None] * len(tiff_files)

    for i, obj in enumerate(tiff_files):
        with tifffile.TiffFile(obj) as t:
            slices[i] = len(t.pages)
            page = t.pages[0]
            width[i] = _get_tiff_tag(page, tag_name="ImageWidth")
            length[i] = _get_tiff_tag(page, tag_name="ImageLength")

    return tiff_files, sizes, slices, width, length


def natural_sort(list_to_sort: list) -> list:
    """
    Sort a list of strings or  in alphanumeric order.

    Parameters
        list_to_sort: A list of strings or pathlib.Path objects to sort naturally

    Returns
        list: A list of strings in alphanumeric order
    """

    sorted_list = sorted(list_to_sort,
                         key=lambda item: [int(part) if part.isdigit() else part.lower()
                                           for part in re.split(r'(\d+)', str(item))])

    if isinstance(list_to_sort[0], pathlib.Path):
        sorted_list = [pathlib.Path(i) for i in sorted_list]

    return sorted_list


def combine_slices(filepath: pathlib.Path, filenames: list[pathlib.Path], use_compression='zlib') -> np.ndarray:
    """
    Combine individual slices into a volumetric stack. To control which slices to include, supply a list of filenames.

    Parameters:
        filepath (str): Path to the directory containing the tiff files
        filenames (list): List of the file names of the tiff files to combine
        use_compression (str): Compression type to use for the combined tiff file

    Returns:
        np.ndarray: Array containing the combined stack
    """

    # Read first slices and determine datatype
    first_slice = read_image(filepath / filenames[0])
    datatype = first_slice.dtype

    # Create new array for combined file
    combined_stack = np.zeros(
        [len(filenames), first_slice.shape[0], first_slice.shape[1]], dtype=datatype
    )

    # Add first slice to array
    combined_stack[0] = np.array(first_slice)

    # Read each image and add to array
    for count, file in enumerate(filenames[1:], 1):
        next_file = read_image(filepath / file)
        combined_stack[count] = np.array(next_file)

    # Convert array to .tiff file and save it
    print("Final shape of combined stack = ", combined_stack.shape)
    print("-" * 53)

    write_image(save_path=filepath, save_name=f'combined_stack_0-{len(filenames)}.tif',
                image=combined_stack, filetype='tiff', compression_type=use_compression,
                tiffSize=(combined_stack.nbytes >= 4294967296))

    return combined_stack


def convert_filetype(filepath: pathlib.Path, convert_to: str, **kwargs) -> None:
    """
    Convert the file extension from one format to another

    Parameters:
        filepath: pathlib.Path object with the path to the file to convert
        convert_to: str extension of the file format to convert to
        **kwargs: keyword arguments needed to read the file

    Returns:
        None
    """
    conversion_list = ['.raw', '.tiff', '.tif', '.nc']

    # Make sure there is no .
    convert_to = convert_to.lower()
    if not convert_to.startswith("."):
        convert_to = "." + convert_to

    original_image = read_image(read_path=filepath, **kwargs)

    new_filepath = filepath.with_suffix(convert_to)
    write_image(save_path=new_filepath.parent, save_name=new_filepath.name, image=original_image, filetype=convert_to)


