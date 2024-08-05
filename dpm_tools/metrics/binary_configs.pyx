# distutils: language = c++
# cython: boundscheck=False, wraparound=False, cdivision=True, nonecheck=False

from libc.limits cimport USHRT_MAX
import numpy as np
cimport numpy as cnp
from _minkowski_coeff import *
from libc.stdlib cimport malloc, free
from cython.parallel import prange

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


    for x in prange(dim0 - 1, nogil=True):
        for y in range(dim1 - 1):
            mask_val = ((get_voxel_value_2d(x, y, image_ptr, dim1) == 1) << 0) \
                   + ((get_voxel_value_2d(x + 1, y, image_ptr, dim1) == 1) << 1) \
                   + ((get_voxel_value_2d(x, y + 1, image_ptr, dim1) == 1) << 2) \
                   + ((get_voxel_value_2d(x + 1, y + 1, image_ptr, dim1) == 1) << 3)

            mask[x, y] = IC_5[mask_val]

    return mask

cpdef cnp.ndarray[cnp.uint64_t, ndim=1, mode="c"] get_configs_histogram_2d(cnp.ndarray[cnp.uint8_t, ndim=2] image, int dim0, int dim1):
    cdef unsigned char mask_val
    cdef cnp.ndarray[cnp.uint64_t, ndim=1, mode="c"] config_hist = np.zeros(6, dtype=np.uint64)
    cdef int x, y, i
    image = np.ascontiguousarray(image)
    cdef unsigned char* image_ptr = <unsigned char*>image.data

    for x in prange(dim0 - 1, nogil=True):
        for y in range(dim1 - 1):
            mask_val = ((get_voxel_value_2d(x, y, image_ptr, dim1) == 1) << 0) \
                   + ((get_voxel_value_2d(x + 1, y, image_ptr, dim1) == 1) << 1) \
                   + ((get_voxel_value_2d(x, y + 1, image_ptr, dim1) == 1) << 2) \
                   + ((get_voxel_value_2d(x + 1, y + 1, image_ptr, dim1) == 1) << 3)

            config_hist[IC_5[mask_val]] += 1


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

                mask[x, y, z] = IC_22[mask_val]

    return mask

cpdef cnp.ndarray[cnp.uint64_t, ndim=1, mode="c"] get_configs_histogram_3d(cnp.ndarray[cnp.uint8_t, ndim=3] image, int dim0, int dim1, int dim2):
    cdef unsigned char mask_val
    cdef cnp.ndarray[cnp.uint64_t, ndim=1, mode="c"] config_hist = np.zeros(22, dtype=np.uint64)
    cdef int x, y, z, i
    image = np.ascontiguousarray(image)
    cdef unsigned char* image_ptr = <unsigned char*>image.data

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

                config_hist[IC_22[mask_val]] += 1

    return config_hist