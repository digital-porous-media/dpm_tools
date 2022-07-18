<<<<<<< HEAD
from dpm_tools.io import Image, convert_filetype
from dpm_tools.visualization import hist, plot_slice, make_thumbnail, make_gif
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as anim
=======
import sys
sys.path.append('C:\\Users\\Frieda\\AppData\\Local\\Programs\\Python\\Python310\\dpm_tools') #Add custom filepath here
from dpm_tools.io import read_image
from dpm_tools.visualization import hist

>>>>>>> 4650681bc03980fcfcc355b78030206e8770cea3


image_info = {
    'bits': 8,
    'signed': 'unsigned',
    'byte_order': 'little',
    'nx': 110,
    'ny': 110,
    'nz': 110,
}
# convert_filetype(filepath='../data/berea_pore.raw', convert_to='tif', meta=image_info)
img = Image(basepath='../data/', filename='berea_pore.raw', meta=image_info)

# for i in range(0, 110, 5):
# plot_slice(img, slice_num=10, cmap='gray')
# make_thumbnail(img)

make_gif(img)
#
# plt.show()
# img = read_image(read_path='../data/35_1.tiff')
# my_hist = hist(img.image, write_csv=False)
