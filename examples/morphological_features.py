import numpy as np
from dpm_tools.metrics import morph_drain, heterogeneity_curve, minkowski_3d
from dpm_tools.io import ImageFromFile
from dpm_tools.visualization import plot_heterogeneity_curve
import matplotlib.pyplot as plt

if __name__ == '__main__':
    # Converting the image to "Image" class
    img = ImageFromFile('data', 'berea_pore.tif')
    img.image = -1 * img.image + 1
    # img.image = edt(img)
    plt.figure(1)
    plt.imshow(img.image[:, img.ny // 2, :], cmap="inferno")
    plt.colorbar()
    plt.show()


    # fluid_configs, sw = _morph_drain_config(img, radius=4.5)
    radii, sw, fluid_configs = morph_drain(img.image, target_saturation=0.1)

    plt.figure(2)
    # plt.subplot(1, 2, 1)
    plt.title(f'Drainage: Rcrit= {radii[-1]:.3f} Sw= {sw[-1]:.3f}')
    plt.pcolormesh(fluid_configs[:, img.ny // 2, :], cmap='hot')
    plt.gca().invert_yaxis()
    # plt.pcolormesh(drain_config[20,:,:],cmap='hot')
    plt.axis('equal')
    plt.grid(True)
    plt.show()

    plt.figure(3)
    plt.plot(sw, radii,  'b-', markersize=6, label='Drainage')
    plt.plot([0, 1], [5, 5], 'r--')
    plt.legend(loc='best')
    plt.grid(True)
    # plt.xlim([0, 1.0])
    # plt.ylim([0, 2.0])

    radii, variances = heterogeneity_curve(img.image)
    plot_heterogeneity_curve(radii, variances)

    plt.show()

    Vn, An, Sn, Xn = minkowski_3d(img.image)
    print("Minkowski Functionals:")
    print(f"\tVolume: {Vn:.3f}\n\tSurface Area: {An:.3}\n\tMean Curvature: {Sn:.3f}\n\tEuler Characteristic: {Xn:.3}")