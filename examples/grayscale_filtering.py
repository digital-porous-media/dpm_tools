from dpm_tools.io import download_file
from dpm_tools.metrics import histogram_statistics as hist_stats
import skimage.io
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    file_url1 = "https://www.digitalrocksportal.org/projects/211/images/126114/download/";
    filename1 = "Project211_sandstone_slice_0801hr.png"
    download_file(file_url1, filename1)

    # image location, this is a lower resolution image (see project)
    file_url2 = "https://www.digitalrocksportal.org/projects/211/images/126113/download/"
    filename2 = "Project211_sandstone_slice_0801lr.png"
    download_file(file_url2, filename2)
    # CHANGE MADE - using one of the three channels
    img_hr = skimage.io.imread(filename1)
    img_hr = img_hr[:,:,0] #use only one channel of the three

    img_lr = skimage.io.imread(filename2)
    img_lr = img_lr[:,:,0] #use only one channel of the three

    f = plt.figure(figsize=(24, 24))
    f.add_subplot(221)
    plt.imshow(img_hr) # if the image has three channels, default - and fake - color will appear when using imshow. Color scheme can easily be changed.
    plt.title('higher res',fontsize=20)

    f.add_subplot(222)
    plt.imshow(img_lr) # if the image has three channels, default - and fake - color will appear when using imshow
    plt.title('lower res',fontsize=20)

    # location and length of an image subset in x, y directions
    locx = 470
    lenx = 90
    locy = 440
    leny = 90

    f.add_subplot(223)
    plt.imshow(img_hr[locx:(locx+lenx),locy:(locy+leny)])
    plt.title('zoom in - higher res',fontsize=20)

    f.add_subplot(224)
    # note that // is floor division (so we get integers as a result)
    plt.imshow(img_lr[locx // 4:(locx+lenx)//4,locy//4:(locy+leny)//4]) 
    plt.title('zoom in - lower res', fontsize=20)
    
    hist_stats(img_hr, img_lr, plot_histogram=True, nbins=128, legend_elem=["img_hr", "img_lr"])
    plt.show()
        
    
