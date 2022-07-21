import glob
import os
import sys
import tifffile as tiff
from PIL import Image
import exifread
import numpy as np
import pandas as pd

from .read_data import read_image
from .write_data import write_image


def _find_files(directory: str, extension: str) -> list:
    sizes = []
    
    path = directory+"/**/*"+extension

    found = glob.glob(path, recursive=True)

    count = 0

    for obj in found:
        count = count + 1
        size = str(os.path.getsize(obj)) + " bytes"
        sizes.append(size)

    found_tuple = list(zip(found, sizes))

    print("There are",count,"files with the",extension,"extension in the directory",directory)
    files_df = pd.DataFrame(found_tuple, columns=['File','Size'])
    print(files_df)
    
    return found_tuple


def _find_tiff_files(directory: str) -> list:
    #Create lists for data columns
    found = []
    files = []
    sizes = []
    folders = []
    slices = []
    width = []
    height = []
    dt = []
    bt = []

    #Find all .tiff files
    extension = ".tiff"
    path = directory+"/**/*"+extension
    found.extend(glob.glob(path, recursive=True))

    #Find all .tif files
    extension = ".tif"
    path = directory+"/**/*"+extension
    found.extend(glob.glob(path, recursive=True))

    #Iterate through each found file
    for obj in found:
        get_folder = obj.split("\\")
        folder_name = ""
        for fold in get_folder:
            if(extension not in fold):
                folder_name = folder_name + "\\" + fold
            elif(extension in fold):
                files.append(fold)
                folders.append(folder_name)
                image = tiff.imread(obj)

                if(len(image.shape) == 2):
                    slices.append(1)
                    width.append(image.shape[0])
                    height.append(image.shape[1])
                else:
                    slices.append(image.shape[0])
                    width.append(image.shape[1])
                    height.append(image.shape[2])
                dt.append(image.dtype)
                bt.append(image.dtype.byteorder)
        size = str(os.path.getsize(obj)) + " bytes"
        sizes.append(size)

    found_tuple = list(zip(files, sizes, folders, slices, width, height, dt, bt))

    files_df = pd.DataFrame(found_tuple, columns=['File', 'Size', 'Folder', 'Slices', 'Width', 'Height', 'Data Type', 'Byte Order'])
    print(files_df)
    
    return found_tuple


#Check if .tiff file is a 2D or 3D image
def _evaluate_dimensions(directory: str, starting_file: str) -> int:
    #Exifread code from https://stackoverflow.com/questions/46477712/reading-tiff-image-metadata-in-python
    path = directory+"\\"+starting_file
    f = open(path, 'rb')

    # Return Exif tags
    tags = exifread.process_file(f)
    tags_list = list(tags.keys())

    #Iterate through each image tag
    i = 0
    slices = 1
    while(slices == 1 and i < len(tags_list)):
        tag = tags_list[i]
        
        #Find image description tag
        if "ImageDescription" in tag:
            value = str(tags[tag])
            description_parts = value.split("\n")

            #Iterate through each part of the description
            for part in description_parts:

                #Find number of slices
                if "slices" in part:
                    find_slices = part.split("=")
                    slices = int(find_slices[-1])
        i += 1

    return slices


def _sort_files(directory: str, extension: str, starting_file: str, slices: int) -> list:
    """
    A function to sort the files in a directory by number slice.
    This is useful when dealing with directories of tiff slices rather than volumetric tiff
    """
    unsorted_files = {}
    sorted_files = []
    count = slices

    # Find all files with extension in directory
    path = directory+"/*"+extension
    found = glob.glob(path)

    # Split up file names for sorting
    for obj in found:
        split1 = obj.split(".")
        split2 = split1[0].split(" ")
        split3 = split2[-1].split("_")
        split4 = split3[-1].split("\\")
        unsorted_files[split4[-1]] = obj

    # Sort files
    sorting_list = sorted(unsorted_files)

    # Append full path names to sorted list using sorted file names
    for i in sorting_list:
        print(i)
        # Start appending names to list using user-provided range
        if i in starting_file:
            sorted_files.append(unsorted_files[i])
            count = 1
        elif count < slices:
            sorted_files.append(unsorted_files[i])
            count = count + 1
    print(sorted_files)

    return sorted_files


# TODO Add option to combine based on indices of desired slices

def _combine_slices(filepath: str, filenames: list, use_compression='zlib') -> np.ndarray:
    
    #Combines individual slices in a stack.
    #To control which slices to include, supply a list of filenames
    

    # Read first slices and determine datatype
    first_slice = read_image(os.path.join(filepath, filenames[0]))
    datatype = first_slice.dtype

    # Create new array for combined file
    combined_stack = np.zeros(
        [len(filenames), first_slice.shape[0], first_slice.shape[1]], dtype=datatype
    )

    # Add first slice to array
    combined_stack[0] = np.array(first_slice)

    # Read each image and add to array
    for count, file in enumerate(filenames[1:], 1):
        next_file = read_image(os.path.join(filepath, file))
        combined_stack[count] = np.array(next_file)

    # Convert array to .tiff file and save it
    print("Final shape of combined stack = ", combined_stack.shape)
    print("-" * 53)
    # Check if bigtiff is needed
    # is_bigtiff = False if combined_stack.nbytes < 4294967296 else True
    
    if(combined_stack.nbytes >= 4294967296):
        write_image(save_path=filepath, save_name=f'combined_stack_0-{len(filenames)}.tif',
                image=combined_stack, filetype='tiff', compression_type=use_compression, tiffSize = True)
    else:
        write_image(save_path=filepath, save_name=f'combined_stack_0-{len(filenames)}.tif',
                image=combined_stack, filetype='tiff', compression_type=use_compression, tiffSize = False)

    #write_image(save_path=filepath, save_name=f'combined_stack_0-{len(filenames)}.tif',
    #            image=combined_stack, filetype='tiff')

    return combined_stack
"""

def _combine_slices(filepath, filenames, substack_name, use_compression='zlib') -> np.ndarray:

    #Read first slices and determine datatype
    first_slice = read_image(os.path.join(filepath, filenames[0]))
    datatype = first_slice.dtype

    #Create new array for combined file
    combined_stack = np.zeros(
        [len(filenames), first_slice.shape[0], first_slice.shape[1]], dtype=datatype
    )

    #Add first slice to array
    combined_stack[0] = np.array(first_slice)
    
    #Read each image and add to array
    for count, file in enumerate(filenames[1:], 1):
        next_file = read_image(os.path.join(filepath, file))
        combined_stack[count] = np.array(next_file)
    
    
    #Convert array to .tiff file and save it
    print("Final shape of combined stack = ", combined_stack.shape)
    print("-"*53)
    if(combined_stack.nbytes >= 4294967296):
        write_image(save_path=filepath, save_name=f'combined_stack_0-{len(filenames)}.tif',
                image=combined_stack, filetype='tiff', compression_type=use_compression, tiffSize=True)
    else:
        write_image(save_path=filepath, save_name=f'combined_stack_0-{len(filenames)}.tif',
<<<<<<< HEAD
                image=combined_stack, filetype='tiff', compression_type=use_compression, tiffSize=False)



=======
                image=combined_stack, filetype='tiff', compression_type=use_compression, tiffSize = False)
"""
>>>>>>> 427a54fb8a015ad3be7b636ec357e470d36c2351


def convert_filetype(filepath: str, convert_to: str, **kwargs) -> None:

    conversion_list = ['raw', 'tiff', 'tif', 'nc']
    filepath = filepath.replace('\\', '/')
    original_image = read_image(read_path=filepath, **kwargs)

    filepath, filename = filepath.rsplit('/', 1)
    basename, extension = filename.rsplit('.', 1)

    assert extension in conversion_list, "Unsupported filetype, cannot convert"

    filename = basename + "." + convert_to.lower()
    write_image(save_path=filepath, save_name=filename, image=original_image, filetype=convert_to)



