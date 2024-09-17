import numba
from numba import jit, prange, cuda
import numpy as np

@jit(nopython=True)
def initialize_mapping(n_dim):
    if n_dim == 2:
        IC = np.array([0, 1, 1, 2, 1, 2, 3, 4, 1, 3, 2, 4, 2, 4, 4, 5])
    elif n_dim == 3:
        IC  = np.array([0, 1, 1, 2, 1, 2, 3, 5, 1, 3,
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
                        19, 20, 19, 20, 20, 21])
    else:
        raise ValueError("Dimensions must be 2 or 3")

    return IC

@jit('u1[:, :](u1[:, :], u8, u8)', nopython=True, parallel=False)
def get_binary_configs_2d(image, dim0, dim1):
    # n_threads = numba.get_num_threads()
    # chunk_size = n_threads * 8
    # n_chunks = (dim0 // (chunk_size)) + 1

    mask = np.zeros((dim0 - 1, dim1 - 1), dtype=np.uint8)

    IC = initialize_mapping(2)

    # for chunk in prange(n_chunks):
    #     start_x = chunk * chunk_size
    #     end_x = min(start_x + chunk_size + 2, dim0)
    for x in range(dim0 - 1):
        for y in range(dim1 - 1):
            mask_val = (int(image[x, y] == 1) +
                        int(image[x + 1, y] == 1) * 2 +
                        int(image[x, y + 1] == 1) * 4 +
                        int(image[x + 1, y + 1] == 1) * 8)
            mask[x, y] = IC[mask_val]

    return mask

@jit('u8[:](u1[:, :], u8, u8)', nopython=True, parallel=True)
def get_configs_histogram_2d(image, dim0, dim1):
    n_threads = numba.get_num_threads()
    thread_hist = np.zeros((n_threads, 6), dtype=np.uint64)
    IC = initialize_mapping(2)

    for x in prange(dim0 - 1):
        thread_id = numba.get_thread_id()
        for y in range(dim1 - 1):
            mask_val = (int(image[x, y] == 1) +
                        int(image[x + 1, y] == 1) * 2 +
                        int(image[x, y + 1] == 1) * 4 +
                        int(image[x + 1, y + 1] == 1) * 8)
            thread_hist[thread_id, IC[mask_val]] += 1

    config_hist = np.sum(thread_hist, axis=0)

    return config_hist
@jit('u1[:, :, :](u1[:, :, :], u8, u8, u8)', nopython=True, parallel=True)
def get_binary_configs_3d(image, dim0, dim1, dim2):
    n_threads = numba.get_num_threads()
    n_chunks = (dim0 // n_threads) + 1

    mask = np.zeros((dim0-1, dim1-1, dim2-1), dtype=np.uint8)
    IC = initialize_mapping(3)

    for chunk in prange(n_chunks):
        start_x = chunk * n_threads
        end_x = min(start_x + n_threads, dim0 - 1)
        for x in range(start_x, end_x):
            for y in range(dim1 - 1):
                for z in range(dim2 - 1):
                    mask_val = (int(image[x, y, z] == 1) +
                                int(image[x + 1, y, z] == 1) * 2 +
                                int(image[x, y + 1, z] == 1) * 4 +
                                int(image[x + 1, y + 1, z] == 1) * 8 +
                                int(image[x, y, z + 1] == 1) * 16 +
                                int(image[x + 1, y, z + 1] == 1) * 32 +
                                int(image[x, y + 1, z + 1] == 1) * 64 +
                                int(image[x + 1, y + 1, z + 1] == 1) * 128)
                    mask[x, y, z] = IC[mask_val]
    return mask

@jit('u8[:](u1[:, :, :], u8, u8, u8)', nopython=True, parallel=True)
def get_configs_histogram_3d(image, dim0, dim1, dim2):
    n_threads = numba.get_num_threads()
    thread_hist = np.zeros((n_threads, 22), dtype=np.uint64)
    IC = initialize_mapping(3)
    for x in prange(dim0 - 1):
        thread_id = numba.get_thread_id()
        for y in range(dim1 - 1):
            for z in range(dim2 - 1):
                mask_val = (int(image[x, y, z] == 1) +
                            int(image[x + 1, y, z] == 1) * 2 +
                            int(image[x, y + 1, z] == 1) * 4 +
                            int(image[x + 1, y + 1, z] == 1) * 8 +
                            int(image[x, y, z + 1] == 1) * 16 +
                            int(image[x + 1, y, z + 1] == 1) * 32 +
                            int(image[x, y + 1, z + 1] == 1) * 64 +
                            int(image[x + 1, y + 1, z + 1] == 1) * 128)

                thread_hist[thread_id, IC[mask_val]] += 1
    config_hist = np.sum(thread_hist, axis=0)

    return config_hist
