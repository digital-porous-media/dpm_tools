from dpm_tools.io import Image, convert_filetype
from dpm_tools.visualization import hist, plot_slice, make_thumbnail, make_gif


image_info = {
    'bits': 8,
    'signed': 'unsigned',
    'byte_order': 'little',
    'nx': 110,
    'ny': 110,
    'nz': 110,
}
convert_filetype(filepath='../data/berea_pore.raw', convert_to='tif', meta=image_info)
img = Image(basepath='../data/', filename='berea_pore.tif')

# plot_slice(img, slice_num=25, cmap='gray')
# make_thumbnail(img)

# TODO something wrong with using raw but ok using tiff
make_gif(img)
# img = read_image(read_path='../data/35_1.tiff')
# my_hist = hist(img.image, write_csv=False)
