import numpy as np
from dpm_tools.metrics import morphological_drainage_3d, edt
from dpm_tools.io import ImageFromFile
import matplotlib.pyplot as plt

if __name__ == '__main__':
    # Converting the image to "Image" class
    img = ImageFromFile('../data', 'berea_pore.tif')
    img.image = -1 * img.image + 1
    # img.image = edt(img)
    plt.figure()
    plt.imshow(img.image[50], cmap="inferno")
    plt.colorbar()
    plt.show()


    # Set initial radius based on the largest Euclidean distance
    Rcrit_new = np.amax(edt(img))
    Rcrit_old = np.amax(edt(img))
    print(Rcrit_new)
    Sw_drain = []
    radii = []
    total_global = np.count_nonzero(img.image)
    target_saturation = 0.1
    # void_fraction = total_global / img.image.size
    saturation_new = 1.0
    saturation_old = 1.0
    saturation_diff_new = 1.0
    saturation_diff_old = 1.0

    # drain_config, saturation_new = morphological_drainage_3d(img, R_critical=Rcrit_new)


    while saturation_new > target_saturation and Rcrit_new > 0.5:
        saturation_diff_old = saturation_diff_new
        saturation_old = saturation_new
        Rcrit_old = Rcrit_new
        Rcrit_new -= round(0.05*Rcrit_old)
        # Generating the morphological drain configuration
        drain_config, saturation_new = morphological_drainage_3d(img, R_critical=Rcrit_new)
        saturation_diff_new = abs(saturation_new - target_saturation)
        Sw_drain.append(saturation_new)
        radii.append(Rcrit_new)

        if saturation_diff_new < saturation_diff_old:
            final_void_fraction = saturation_new
        else:
            final_void_fraction = saturation_old

        print(f"{final_void_fraction}, {Rcrit_new}")


    plt.figure(1)
    # plt.subplot(1, 2, 1)
    plt.title('Drainage: Rcrit=' + str(Rcrit_new) + ' Sw=' + str(saturation_new))
    plt.pcolormesh(drain_config[:, :, img.nx // 2], cmap='hot')
    # plt.pcolormesh(drain_config[20,:,:],cmap='hot')
    plt.axis('equal')
    plt.grid(True)
    plt.show()

    plt.figure(2)
    plt.plot(Sw_drain, radii,  'ro', markersize=6, label='Drainage')
    plt.legend(loc='best')
    plt.grid(True)
    # plt.xlim([0, 1.0])
    # plt.ylim([0, 2.0])

    plt.show()