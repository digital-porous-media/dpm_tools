from dpm_tools.visualization import plot_isosurface
from dpm_tools.io import Image
import pathlib
import numpy as np
import matplotlib.pyplot as plt
import imageio
from tqdm import tqdm


if __name__ == '__main__':
    filenames = []
    for i in tqdm(range(5, 96, 5)):
        dir_path = pathlib.Path(rf"C:\Users\bchan\Box\morphdrain_castlegate\sw_{i}.raw")
        filenames.append(f"{dir_path.parent}/vis_dir/{dir_path.stem}.png")
        lrc32 = np.fromfile(dir_path, dtype=np.uint8).reshape([512, 512, 512])
        lrc32 = lrc32[128:300, 128:300, 128:300]
        lrc32 = np.pad(lrc32, pad_width=1, mode='constant')
        img = Image(scalar=lrc32)

        fig = plot_isosurface(img, show_isosurface=[0.5], mesh_kwargs={'opacity': 0.05, 'color': (94, 237, 225)})
        img.scalar = np.pad(img.scalar, pad_width=1, mode='constant', constant_values=1)
        fig = plot_isosurface(img, fig=fig, show_isosurface=[1.5], mesh_kwargs={'opacity': 1, 'color': (149, 172, 237)})
        fig.show(cpos=[(445.4367341839443, 468.84171565685955, 332.86221745394596),
                        (95.53319355749468, 82.92730167950663, 80.41970280798188),
                        (-0.31230647741936785, -0.305293986406427, 0.899588931693792)],
                 screenshot=f"{dir_path.parent}/vis_dir/{dir_path.stem}.png",
                 interactive=False)


    images = []
    for filename in filenames[::-1]:
        images.append(imageio.imread(filename))
    imageio.mimsave(f'{dir_path.parent}/vis_dir/morphdrain.gif', images, fps=5)
