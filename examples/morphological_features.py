import numpy as np
from dpm_tools.metrics import morph_drain, heterogeneity_curve, minkowski_3d
from dpm_tools.io import ImageFromFile
from dpm_tools.visualization import plot_heterogeneity_curve
import matplotlib.pyplot as plt
import pandas as pd
import pathlib
from natsort import natsorted

if __name__ == '__main__':
    parent_path = pathlib.Path(r"C:\Users\bcc2459\Box\Pore Model Image sharing\porous solid bitumen pore models")

    names = []
    heterogeneity_radii = []
    heterogeneity_variances = []
    # Vn = An = Sn = Xn = np.empty((len(list(parent_path.glob('*.raw')))), dtype=np.float64)
    Vn = np.empty((len(list(parent_path.glob('*.raw')))), dtype=np.float64)
    An = np.empty((len(list(parent_path.glob('*.raw')))), dtype=np.float64)
    Sn = np.empty((len(list(parent_path.glob('*.raw')))), dtype=np.float64)
    Xn = np.empty((len(list(parent_path.glob('*.raw')))), dtype=np.float64)
    paths = list(parent_path.glob('*.raw'))
    paths = natsorted(paths)


    for i, file in enumerate(paths):
        print(file.name)
        names.append(file.name)
        # Converting the image to "Image" class
        img = ImageFromFile(parent_path, file.name, meta={'bits': 8, 'byte_order': 'little',
                                                          'nz': 1000, 'ny': 1000, 'nx': 1000,
                                                          'signed': 'signed'})
        img.image = -1 * img.image + 1
        # img.image = edt(img)
        # plt.figure(1)
        # plt.imshow(img.image[:, img.ny // 2, :], cmap="inferno")
        # plt.colorbar()
        # plt.show()
        #
        #
        # # fluid_configs, sw = _morph_drain_config(img, radius=4.5)
        # radii, sw, fluid_configs = morph_drain(img.image, target_saturation=0.1)
        #
        # plt.figure(2)
        # # plt.subplot(1, 2, 1)
        # plt.title(f'Drainage: Rcrit= {radii[-1]:.3f} Sw= {sw[-1]:.3f}')
        # plt.pcolormesh(fluid_configs[:, img.ny // 2, :], cmap='hot')
        # plt.gca().invert_yaxis()
        # # plt.pcolormesh(drain_config[20,:,:],cmap='hot')
        # plt.axis('equal')
        # plt.grid(True)
        # plt.show()
        #
        # plt.figure(3)
        # plt.plot(sw, radii,  'b-', markersize=6, label='Drainage')
        # plt.plot([0, 1], [5, 5], 'r--')
        # plt.legend(loc='best')
        # plt.grid(True)
        # # plt.xlim([0, 1.0])
        # # plt.ylim([0, 2.0])

        radii, variances = heterogeneity_curve(img.image)
        heterogeneity_radii.append(radii)
        heterogeneity_variances.append(variances)
        # plot_heterogeneity_curve(radii, variances)

        # plt.show()

        Vn[i], An[i], Sn[i], Xn[i] = minkowski_3d(img.image)
        print("Minkowski Functionals:")
        print(f"\tVolume: {Vn[i]:.3f}\n\tSurface Area: {An[i]:.3}\n\tMean Curvature: {Sn[i]:.3f}\n\tEuler Characteristic: {Xn[i]:.3}")

    df = pd.DataFrame(data={'Name': names,
                            'Vsi_Radius': heterogeneity_radii, 'Vsi_Variance': heterogeneity_variances,
                            'Vn': Vn, 'An': An, 'Sn': Sn, 'Xn': Xn})

    # df = pd.read_parquet(pathlib.Path(r"C:\Users\bcc2459\Box") / '3d_metrics.parquet')
    # print(df.head())
    # df['Vn'] = Vn
    # df['An'] = An
    # df['Sn'] = Sn
    # df['Xn'] = Xn
    #
    df.to_csv(parent_path.parent.parent / '3d_metrics.csv')
    df.to_parquet(parent_path.parent.parent / '3d_metrics.parquet')
    print(df.describe())