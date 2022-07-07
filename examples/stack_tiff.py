from dpm_tools.io import Image
from dpm_tools.visualization import hist, plot_slice
import os
import sys
print(sys.path)


image_info = {
    'bits': 8,
    'signed': 'unsigned',
    'byte_order': 'little',
    'nx': 110,
    'ny': 110,
    'nz': 110,
}
# convert_filetype(filepath='../data/3_fractures.tif', convert_to='raw')#, metadata_dict=image_info)
img = Image(basepath='../data/', filename='35_1.tiff')

plot_slice(img, cmap='gray')
# img = read_image(read_path='../data/35_1.tiff')
# my_hist = hist(img.image, write_csv=False)
