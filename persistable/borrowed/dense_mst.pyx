# cython: boundscheck=False, wraparound=False, cdivision=True

# Copyright (c) 2001-2002 Enthought, Inc. 2003-2022, SciPy Developers.
#
# Function to compute stepwise dendrogram of a hierarchy given by a distance
# matrix plus core distances. modified to be used in Persistable pipeline.
# Originally in scipy/scipy/cluster/_hierarchy.pyx in
# https://github.com/scipy/scipy/commit/2f4bd4234370adcd7085bcc145f9c302f231408f

import numpy as np
cimport numpy as np


cdef extern from "numpy/npy_math.h":
    cdef enum:
        NPY_INFINITYF

def stepwise_dendrogram_with_core_distances(int n, const double[:,:] dists, const double[:] core_distance):
    Z_arr = np.empty((n - 1, 3))
    cdef double[:, :] Z = Z_arr

    # Which nodes were already merged.
    cdef int[:] merged = np.zeros(n, dtype=np.intc)

    cdef double[:] D = np.empty(n)
    D[:] = NPY_INFINITYF

    cdef int i, k, x, y = 0
    cdef double dist, current_min

    cdef np.ndarray result_arr 
    cdef np.ndarray core_distance_arr
    cdef np.double_t *core_distance_ptr

    core_distance_arr = np.asarray(core_distance,dtype=np.double)
    cdef np.double_t[::1] core_distance_ = (<np.double_t[:n:1]> (
        <np.double_t *> core_distance_arr.data))

    core_distance_ptr = <np.double_t *> &core_distance_[0]

    x = 0
    with nogil:
        for k in range(n - 1):
            current_min = NPY_INFINITYF
            merged[x] = 1
            for i in range(n):
                if merged[i] == 1:
                    continue

                #dist = max(dists[condensed_index(n, x, i)],core_distances[x],core_distances[i])
                dist = max(dists[x, i],core_distance_ptr[x],core_distance_ptr[i])
                if D[i] > dist:
                    D[i] = dist

                if D[i] < current_min:
                    y = i
                    current_min = D[i]

            Z[k, 0] = x
            Z[k, 1] = y
            Z[k, 2] = current_min
            x = y

    # Sort Z by cluster distances.
    order = np.argsort(Z_arr[:, 2], kind='mergesort')
    Z_arr = Z_arr[order]

    # do not relabel since we don't really need it, and this makes things easier
    # when combining different hierarchical clusterings
    # Find correct cluster labels and compute cluster sizes inplace.
    #label(Z_arr, n, n)

    return Z_arr