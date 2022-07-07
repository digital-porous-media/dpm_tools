import sys
sys.path.append('C:\\Users\\Frieda\\AppData\\Local\\Programs\\Python\\Python310\\dpm_tools') #Add custom filepath here
from dpm_tools.io import read_image
from dpm_tools.visualization import hist



image_info = {
    'bits': 8,
    'signed': 'unsigned',
    'byte_order': 'little',
    'nx': 110,
    'ny': 110,
    'nz': 110,
}
# convert_filetype(filepath='../data/3_fractures.tif', convert_to='raw')#, metadata_dict=image_info)
img = read_image(read_path='../data/35_1.tiff')
my_hist = hist(img, write_csv=True)
