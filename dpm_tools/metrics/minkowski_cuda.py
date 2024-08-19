# import numba
# from numba import jit, prange, cuda
# import numpy as np
#
# @jit(nopython=True)
# def initialize_mapping(n_dim):
#     if n_dim == 2:
#         IC = np.array([0, 1, 1, 2, 1, 2, 3, 4, 1, 3, 2, 4, 2, 4, 4, 5])
#     elif n_dim == 3:
#         IC  = np.array([0, 1, 1, 2, 1, 2, 3, 5, 1, 3,
#                         2, 5, 2, 5, 5, 8, 1, 2, 3, 5,
#                         3, 5, 7, 9, 4, 6, 6, 10, 6, 10,
#                         11, 16, 1, 3, 2, 5, 4, 6, 6, 10,
#                         3, 7, 5, 9, 6, 11, 10, 16, 2, 5,
#                         5, 8, 6, 10, 11, 16, 6, 11, 10, 16,
#                         12, 15, 15, 19, 1, 3, 4, 6, 2, 5,
#                         6, 10, 3, 7, 6, 11, 5, 9, 10, 16,
#                         2, 5, 6, 10, 5, 8, 11, 16, 6, 11,
#                         12, 15, 10, 16, 15, 19, 3, 7, 6, 11,
#                         6, 11, 12, 15, 7, 13, 11, 14, 11, 14,
#                         15, 18, 5, 9, 10, 16, 10, 16, 15, 19,
#                         11, 14, 15, 18, 15, 18, 17, 20, 1, 4,
#                         3, 6, 3, 6, 7, 11, 2, 6, 5, 10,
#                         5, 10, 9, 16, 3, 6, 7, 11, 7, 11,
#                         13, 14, 6, 12, 11, 15, 11, 15, 14, 18,
#                         2, 6, 5, 10, 6, 12, 11, 15, 5, 11,
#                         8, 16, 10, 15, 16, 19, 5, 10, 9, 16,
#                         11, 15, 14, 18, 10, 15, 16, 19, 15, 17,
#                         18, 20, 2, 6, 6, 12, 5, 10, 11, 15,
#                         5, 11, 10, 15, 8, 16, 16, 19, 5, 10,
#                         11, 15, 9, 16, 14, 18, 10, 15, 15, 17,
#                         16, 19, 18, 20, 5, 11, 10, 15, 10, 15,
#                         15, 17, 9, 14, 16, 18, 16, 18, 19, 20,
#                         8, 16, 16, 19, 16, 19, 18, 20, 16, 18,
#                         19, 20, 19, 20, 20, 21])
#     else:
#         raise ValueError("Dimensions must be 2 or 3")
#
#     return IC
#
# @jit(nopython=True)
# def get_voxel_value_2d(x, y, image, dim1):
#     i = x*dim1 + y
#     return image[i]
#
# @jit(nopython=True, parallel=True)
# def get_binary_configs_2d(image, dim0, dim1):
#     mask = np.zeros((dim0-1, dim1-1), dtype=np.uint8)
#
#     IC = initialize_mapping(2)
#
#     for x in prange(dim0 - 1):
#         for y in range(dim1 - 1):
#             mask_val = ((get_voxel_value_2d(x, y, image, dim1) == 1) << 0) \
#                    + ((get_voxel_value_2d(x + 1, y, image, dim1) == 1) << 1) \
#                    + ((get_voxel_value_2d(x, y + 1, image, dim1) == 1) << 2) \
#                    + ((get_voxel_value_2d(x + 1, y + 1, image, dim1) == 1) << 3)
#             mask[x, y] = IC[mask_val]
#
#     return mask
#
# @jit(nopython=True, parallel=True)
# def get_configs_histogram_2d(image, dim0, dim1):
#     config_hist = np.zeros(6, dtype=np.uint64)
#     x, y = cuda.grid(2)
#     if x < dim0 - 1 and y < dim1 -1:
#
#     for x in prange(dim0 - 1):
#         thread_id = numba.threading.get_thread_id()
#         for y in range(dim1 - 1):
#             mask_val = (int(image[x, y] == 1) +
#                         int(image[x + 1, y] == 1) * 2 +
#                         int(image[x, y + 1] == 1) * 4 +
#                         int(image[x + 1, y + 1] == 1) * 8)
#             config_hist[IC[mask_val]] += 1
#     return config_hist
#
# @cuda.jit
# def
#
# @jit(nopython=True)
# def get_voxel_value_3d(x, y, z, image, dim1, dim2):
#
#     i = (x*dim1 + y)*dim2 + z
#
#     return image[i]
# @jit(nopython=True, parallel=True)
# def get_binary_configs_3d(image, dim0, dim1, dim2):
#     mask = np.zeros((dim0-1, dim1-1, dim2-1), dtype=np.uint8)
#     IC = initialize_mapping(3)
#
#     for x in prange(dim0 - 1):
#         for y in range(dim1 - 1):
#             for z in range(dim2 - 1):
#                 mask_val = ((get_voxel_value_3d(x, y, z, image, dim1, dim2) == 1) << 0) \
#                        + ((get_voxel_value_3d(x + 1, y, z, image, dim1, dim2) == 1) << 1) \
#                        + ((get_voxel_value_3d(x, y + 1, z, image, dim1, dim2) == 1) << 2) \
#                        + ((get_voxel_value_3d(x + 1, y + 1, z, image, dim1, dim2) == 1) << 3) \
#                        + ((get_voxel_value_3d(x, y, z + 1, image, dim1, dim2) == 1) << 4) \
#                        + ((get_voxel_value_3d(x + 1, y, z + 1, image, dim1, dim2) == 1) << 5) \
#                        + ((get_voxel_value_3d(x, y + 1, z + 1, image, dim1, dim2) == 1) << 6) \
#                        + ((get_voxel_value_3d(x + 1, y + 1, z + 1, image, dim1, dim2) == 1) << 7)
#                 mask[x, y, z] = IC[mask_val]
#     return mask
#
# @jit(nopython=True, parallel=True)
# def get_configs_histogram_3d(image, dim0, dim1, dim2):
#     config_hist = np.zeros(22, dtype=np.uint64)
#     IC = initialize_mapping(3)
#     for x in prange(dim0 - 1):
#         for y in range(dim1 - 1):
#             for z in range(dim2 - 1):
#                 mask_val = ((get_voxel_value_3d(x, y, z, image, dim1, dim2) == 1) << 0) \
#                        + ((get_voxel_value_3d(x + 1, y, z, image, dim1, dim2) == 1) << 1) \
#                        + ((get_voxel_value_3d(x, y + 1, z, image, dim1, dim2) == 1) << 2) \
#                        + ((get_voxel_value_3d(x + 1, y + 1, z, image, dim1, dim2) == 1) << 3) \
#                        + ((get_voxel_value_3d(x, y, z + 1, image, dim1, dim2) == 1) << 4) \
#                        + ((get_voxel_value_3d(x + 1, y, z + 1, image, dim1, dim2) == 1) << 5) \
#                        + ((get_voxel_value_3d(x, y + 1, z + 1, image, dim1, dim2) == 1) << 6) \
#                        + ((get_voxel_value_3d(x + 1, y + 1, z + 1, image, dim1, dim2) == 1) << 7)
#                 config_hist[IC[mask_val]] += 1
#
#     return config_hist