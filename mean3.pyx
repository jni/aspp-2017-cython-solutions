import cython
import numpy as np

@cython.boundscheck(False)
def mean3filter3(double[::1] arr, double[::1] arr_out):
    # make sure arr_out is filled with zeros
    cdef int i, j
    for i in range(1, arr.shape[0]-1):
        for j in range(i-1, i+2):
            arr_out[i] += arr[j]
        arr_out[i] /= 3
    arr_out[0] = (arr[0] + arr[1]) / 2
    arr_out[-1] = (arr[-1] + arr[-2]) / 2
    return np.asarray(arr_out)
