from dpm_tools.io import Image, convert_filetype
from dpm_tools.visualization import hist, plot_slice, make_thumbnail, make_gif
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as anim


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
