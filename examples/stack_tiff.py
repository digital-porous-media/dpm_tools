from dpm_tools.io import convert_filetype


image_info = {
    'bits': 8,
    'signed': 'unsigned',
    'byte_order': 'little',
    'nx': 110,
    'ny': 110,
    'nz': 110,
}
convert_filetype(filepath='../data/3_fractures.tif', convert_to='raw')#, metadata_dict=image_info)

