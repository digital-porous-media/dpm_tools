from dpm_tools.visualization.plot_3d import plot_glyph, bounding_box, plot_streamlines, plot_isosurface, plot_scalar_volume
from dpm_tools.visualization.plot_2d import plot_slice
from dpm_tools.io import Image
import matplotlib.pyplot as plt
import pyvista as pv
import h5py
import numpy as np


h5file = h5py.File(rf"C:\Users\bchan\Box\morphdrain_castlegate\256_vis\00000.h5")['domain_00000']
vel_x = h5file['Velocity_x'][:][120:130]
vel_y = h5file['Velocity_y'][:]
vel_z = h5file['Velocity_z'][:][120:130, :, :]
bin_image = np.pad((vel_z.copy() == 0).astype(np.uint8), pad_width=1, constant_values=0)
# plt.imshow(vel_z[:, 0, :])
# plt.show()

img = Image(scalar=vel_z)
# img.scalar = img.vector[-1].copy()
bin_img = Image(scalar=bin_image)


## Plotting the streamlines with a bounding box
# fig_streamlines = plot_streamlines(img)
fig_streamlines = plot_scalar_volume(img, mesh_kwargs={"opacity": 1})
## Adding a bounding box to the streamlines
# bounding_box(img, fig=fig_streamlines)
plot_isosurface(bin_img, fig=fig_streamlines, show_isosurface=[0.5],
                mesh_kwargs={'color': (210, 207, 214), 'opacity': 1})
# (186, 157, 113)
# inlet_outlet = np.zeros_like(bin_image, dtype=np.float32)
# inlet_outlet = np.pad(inlet_outlet, pad_width=((0, 0), (50, 50), (0, 0)), constant_values=1)
# in_out = Image(scalar=inlet_outlet)
# plot_scalar_volume(in_out, fig=fig_streamlines, origin=(0.0, -100.0, 0.0),
#                    mesh_kwargs={"cmap": "blues", "opacity": 0.25})
# plot_isosurface(in_out, fig=fig_streamlines, show_isosurface=[0.5], origin=(0.0, -25.0, 0.0),
#                 mesh_kwargs={'color': (54, 73, 196), 'opacity': 0.25})
fig_streamlines.show_axes()
cpos = fig_streamlines.show(cpos=[(793.1917826644008, 139.4353353220267, 116.30777662679353),
 (13.581746918718125, 132.76673667991076, 115.40151640137054),
 (-0.0011121418199207676, -0.005879357817294941, 0.9999820979858732)],
                            screenshot=r"C:\Users\bchan\Box\morphdrain_castlegate\256_vis\vfield_1.png",
                            interactive=False)

# print(cpos)
