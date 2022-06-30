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

    # Assign data type based on input
    if (signed.lower() == 'unsigned'): dt1 = 'u'
    
    elif (signed.lower() == 'signed'): dt1 = 'i'
    
    elif (signed.lower() == 'real'): dt1 = 'f'

    dt2 = bits // 8

    #Assign byte order based on input
    if(byte_order.lower() == 'little'):
        bt = '<'
    elif(byte_order.lower() == 'big'):
        bt = '>'
    else:
        bt = '|'

    datatype = bt + dt1 + str(dt2)

    # Load the image into an array and reshape it
    return np.fromfile(filepath, dtype=datatype).reshape([nz, ny, nx])


def _read_nc(filepath: str) -> np.ndarray:

    ds = nc.Dataset(filepath)

    array_name = ""

    #Search metadata variables for the image array name
    for var in ds.variables.values():
        str_var = str(var)
        var_parts = str_var.split("\n")

        #Find the shape variable
        for part in var_parts:
            if "current shape" in part:
                find_numbers = part.split(" ")
                dimension_count = 0

                #Search for arrays with more than 1 dimension
                for num_part in find_numbers:
                    if(num_part.isalpha() == False and num_part not in string.punctuation):
                        dimension_count += 1

        #Extract the name from the correct array
        if dimension_count > 1:
            var_list1 = var_parts[1]
            find_name = var_list1.split(" ")
            i = 0

            #Add the name to a string
            while(find_name[1][i] not in string.punctuation):
                array_name += find_name[1][i]
                i += 1
        
    #Load the image into an array

    image_array = ds[array_name][:]
    
    return image_array

def read_image(read_path: str, **kwargs) -> np.ndarray:
    """
    A general use function for reading in an image of any filetype
    Currently supports reading in tiff and raw.
    """
    filetypes = {'tiff': _read_tiff,
                 'tif': _read_tiff,
                 'raw': _read_raw,
                 'nc': _read_nc}

    filetype = read_path.rsplit('.', 1)[1]

    try:
        return filetypes[filetype.lower()](read_path, kwargs)
    except NotImplemented:
        print('Cannot read supplied filetype yet')
