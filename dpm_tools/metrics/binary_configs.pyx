# distutils: language = c++
# cython: boundscheck=False, wraparound=False, cdivision=True, nonecheck=False

from libc.limits cimport USHRT_MAX
import numpy as np
cimport numpy as cnp
# from _minkowski_coeff import IC_5, IC_22
from libc.stdlib cimport malloc, free
from libc.string cimport memcpy
from cython.parallel import prange

cdef int *IC = NULL
cdef int *IC_5 = NULL
cdef int *IC_22 = NULL

def initialize_mapping(n_dim):
    global IC
    if n_dim == 2:
        IC = <int *> malloc(16 * sizeof(char))
        if IC == NULL:
            raise MemoryError("Failed to allocate memory")
        IC[:] = [0, 1, 1, 2, 1, 2, 3, 4, 1, 3, 2, 4, 2, 4, 4, 5]
    elif n_dim == 3:
        IC = <int *> malloc(256 * sizeof(int))
        if IC == NULL:
            raise MemoryError("Failed to allocate memory")
        IC[:] = [0, 1, 1, 2, 1, 2, 3, 5, 1, 3,
                    2, 5, 2, 5, 5, 8, 1, 2, 3, 5,
                    3, 5, 7, 9, 4, 6, 6, 10, 6, 10,
                    11, 16, 1, 3, 2, 5, 4, 6, 6, 10,
                    3, 7, 5, 9, 6, 11, 10, 16, 2, 5,
                    5, 8, 6, 10, 11, 16, 6, 11, 10, 16,
                    12, 15, 15, 19, 1, 3, 4, 6, 2, 5,
                    6, 10, 3, 7, 6, 11, 5, 9, 10, 16,
                    2, 5, 6, 10, 5, 8, 11, 16, 6, 11,
                    12, 15, 10, 16, 15, 19, 3, 7, 6, 11,
                    6, 11, 12, 15, 7, 13, 11, 14, 11, 14,
                    15, 18, 5, 9, 10, 16, 10, 16, 15, 19,
                    11, 14, 15, 18, 15, 18, 17, 20, 1, 4,
                    3, 6, 3, 6, 7, 11, 2, 6, 5, 10,
                    5, 10, 9, 16, 3, 6, 7, 11, 7, 11,
                    13, 14, 6, 12, 11, 15, 11, 15, 14, 18,
                    2, 6, 5, 10, 6, 12, 11, 15, 5, 11,
                    8, 16, 10, 15, 16, 19, 5, 10, 9, 16,
                    11, 15, 14, 18, 10, 15, 16, 19, 15, 17,
                    18, 20, 2, 6, 6, 12, 5, 10, 11, 15,
                    5, 11, 10, 15, 8, 16, 16, 19, 5, 10,
                    11, 15, 9, 16, 14, 18, 10, 15, 15, 17,
                    16, 19, 18, 20, 5, 11, 10, 15, 10, 15,
                    15, 17, 9, 14, 16, 18, 16, 18, 19, 20,
                    8, 16, 16, 19, 16, 19, 18, 20, 16, 18,
                    19, 20, 19, 20, 20, 21]
    else:
        raise ValueError("Dimensions must be 2 or 3")
# def initialize_2d_mapping():
#     global IC_5
#     IC_5 = <int*>malloc(16 * sizeof(char))
#     if IC_5 == NULL:
#         raise MemoryError("Failed to allocate memory")
#     IC_5[:] = [0, 1, 1, 2, 1, 2, 3, 4, 1, 3, 2, 4, 2, 4, 4, 5]
#
#
# def initialize_3d_mapping():
#     global IC_22
#     IC_22 = <int*>malloc(256 * sizeof(int))
#     if IC_22 == NULL:
#         raise MemoryError("Failed to allocate memory")
#     IC_22[:] =  [0, 1, 1, 2, 1, 2, 3, 5, 1, 3,
#               2, 5, 2, 5, 5, 8, 1, 2, 3, 5,
#               3, 5, 7, 9, 4, 6, 6, 10, 6, 10,
#               11, 16, 1, 3, 2, 5, 4, 6, 6, 10,
#               3, 7, 5, 9, 6, 11, 10, 16, 2, 5,
#               5, 8, 6, 10, 11, 16, 6, 11, 10, 16,
#               12, 15, 15, 19, 1, 3, 4, 6, 2, 5,
#               6, 10, 3, 7, 6, 11, 5, 9, 10, 16,
#               2, 5, 6, 10, 5, 8, 11, 16, 6, 11,
#               12, 15, 10, 16, 15, 19, 3, 7, 6, 11,
#               6, 11, 12, 15, 7, 13, 11, 14, 11, 14,
#               15, 18, 5, 9, 10, 16, 10, 16, 15, 19,
#               11, 14, 15, 18,  15, 18, 17, 20, 1, 4,
#               3, 6, 3, 6, 7, 11, 2, 6, 5, 10,
#               5, 10, 9, 16, 3, 6, 7, 11, 7, 11,
#               13, 14, 6, 12, 11, 15, 11, 15, 14, 18,
#               2, 6, 5, 10, 6, 12, 11, 15, 5, 11,
#               8, 16, 10, 15, 16, 19, 5, 10, 9, 16,
#               11, 15, 14, 18, 10, 15, 16, 19, 15, 17,
#               18, 20, 2, 6, 6, 12, 5, 10, 11, 15,
#               5, 11, 10, 15, 8, 16, 16, 19, 5, 10,
#               11, 15, 9, 16,14, 18, 10, 15, 15, 17,
#               16, 19, 18, 20, 5, 11, 10, 15, 10, 15,
#               15, 17, 9, 14, 16, 18, 16, 18, 19, 20,
#               8, 16, 16, 19, 16, 19, 18, 20, 16, 18,
#               19, 20, 19, 20, 20, 21]

cdef inline unsigned char get_voxel_value_2d(int x, int y, unsigned char* image, int dim1) nogil:
    cdef int i;

    i = x*dim1 + y

    return image[i]

cpdef cnp.ndarray[cnp.uint8_t, ndim=2, mode="c"] get_binary_configs_2d(cnp.ndarray[cnp.uint8_t, ndim=2] image, int dim0, int dim1):
    cdef unsigned char mask_val
    cdef cnp.ndarray[cnp.uint8_t, ndim=2, mode="c"] mask = np.zeros((dim0-1, dim1-1), dtype=np.uint8)

    cdef int x, y, i
    image = np.ascontiguousarray(image)
    cdef unsigned char* image_ptr = <unsigned char*>image.data
    initialize_mapping(2)

    for x in prange(dim0 - 1, nogil=True):
        for y in range(dim1 - 1):
            mask_val = ((get_voxel_value_2d(x, y, image_ptr, dim1) == 1) << 0) \
                   + ((get_voxel_value_2d(x + 1, y, image_ptr, dim1) == 1) << 1) \
                   + ((get_voxel_value_2d(x, y + 1, image_ptr, dim1) == 1) << 2) \
                   + ((get_voxel_value_2d(x + 1, y + 1, image_ptr, dim1) == 1) << 3)
            mask[x, y] = IC[mask_val]
    free(IC)

    return mask

cpdef cnp.ndarray[cnp.uint64_t, ndim=1, mode="c"] get_configs_histogram_2d(cnp.ndarray[cnp.uint8_t, ndim=2] image, int dim0, int dim1):
    cdef unsigned char mask_val
    cdef cnp.ndarray[cnp.uint64_t, ndim=1, mode="c"] config_hist = np.zeros(6, dtype=np.uint64)
    cdef int x, y, i
    image = np.ascontiguousarray(image)
    cdef unsigned char* image_ptr = <unsigned char*>image.data
    initialize_mapping(2)

    for x in prange(dim0 - 1, nogil=True):
        for y in range(dim1 - 1):
            mask_val = ((get_voxel_value_2d(x, y, image_ptr, dim1) == 1) << 0) \
                   + ((get_voxel_value_2d(x + 1, y, image_ptr, dim1) == 1) << 1) \
                   + ((get_voxel_value_2d(x, y + 1, image_ptr, dim1) == 1) << 2) \
                   + ((get_voxel_value_2d(x + 1, y + 1, image_ptr, dim1) == 1) << 3)
            config_hist[IC[mask_val]] += 1
    free(IC)
    return config_hist

cdef inline unsigned char get_voxel_value_3d(int x, int y, int z, unsigned char* image, int dim1, int dim2) nogil:
    cdef int i;

    i = (x*dim1 + y)*dim2 + z

    return image[i]

cpdef cnp.ndarray[cnp.uint8_t, ndim=3, mode="c"] get_binary_configs_3d(cnp.ndarray[cnp.uint8_t, ndim=3] image, int dim0, int dim1, int dim2):
    cdef unsigned char mask_val
    cdef cnp.ndarray[cnp.uint8_t, ndim=3, mode="c"] mask = np.zeros((dim0-1, dim1-1, dim2-1), dtype=np.uint8)
    # cdef cnp.ndarray[cnp.uint8_t, ndim=3, mode="c"] configs
    cdef int x, y, z, i
    image = np.ascontiguousarray(image)
    cdef unsigned char* image_ptr = <unsigned char*>image.data
    initialize_mapping(3)

    for x in prange(dim0 - 1, nogil=True):
        for y in range(dim1 - 1):
            for z in range(dim2 - 1):
                mask_val = ((get_voxel_value_3d(x, y, z, image_ptr, dim1, dim2) == 1) << 0) \
                       + ((get_voxel_value_3d(x + 1, y, z, image_ptr, dim1, dim2) == 1) << 1) \
                       + ((get_voxel_value_3d(x, y + 1, z, image_ptr, dim1, dim2) == 1) << 2) \
                       + ((get_voxel_value_3d(x + 1, y + 1, z, image_ptr, dim1, dim2) == 1) << 3) \
                       + ((get_voxel_value_3d(x, y, z + 1, image_ptr, dim1, dim2) == 1) << 4) \
                       + ((get_voxel_value_3d(x + 1, y, z + 1, image_ptr, dim1, dim2) == 1) << 5) \
                       + ((get_voxel_value_3d(x, y + 1, z + 1, image_ptr, dim1, dim2) == 1) << 6) \
                       + ((get_voxel_value_3d(x + 1, y + 1, z + 1, image_ptr, dim1, dim2) == 1) << 7)
                mask[x, y, z] = IC[mask_val]
    free(IC)
    return mask

cpdef cnp.ndarray[cnp.uint64_t, ndim=1, mode="c"] get_configs_histogram_3d(cnp.ndarray[cnp.uint8_t, ndim=3] image, int dim0, int dim1, int dim2):
    cdef unsigned char mask_val
    cdef cnp.ndarray[cnp.uint64_t, ndim=1, mode="c"] config_hist = np.zeros(22, dtype=np.uint64)
    cdef int x, y, z, i
    image = np.ascontiguousarray(image)
    cdef unsigned char* image_ptr = <unsigned char*>image.data
    initialize_mapping(3)
    for x in prange(dim0 - 1, nogil=True):
        for y in range(dim1 - 1):
            for z in range(dim2 - 1):
                mask_val = ((get_voxel_value_3d(x, y, z, image_ptr, dim1, dim2) == 1) << 0) \
                       + ((get_voxel_value_3d(x + 1, y, z, image_ptr, dim1, dim2) == 1) << 1) \
                       + ((get_voxel_value_3d(x, y + 1, z, image_ptr, dim1, dim2) == 1) << 2) \
                       + ((get_voxel_value_3d(x + 1, y + 1, z, image_ptr, dim1, dim2) == 1) << 3) \
                       + ((get_voxel_value_3d(x, y, z + 1, image_ptr, dim1, dim2) == 1) << 4) \
                       + ((get_voxel_value_3d(x + 1, y, z + 1, image_ptr, dim1, dim2) == 1) << 5) \
                       + ((get_voxel_value_3d(x, y + 1, z + 1, image_ptr, dim1, dim2) == 1) << 6) \
                       + ((get_voxel_value_3d(x + 1, y + 1, z + 1, image_ptr, dim1, dim2) == 1) << 7)
                config_hist[IC[mask_val]] += 1
    free(IC)
    return config_hist