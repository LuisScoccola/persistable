# cython: boundscheck=False, wraparound=False, cdivision=True

# Copyright (c) 2001-2002 Enthought, Inc. 2003-2022, SciPy Developers.
#
# Function to compute stepwise dendrogram of a hierarchy given by a distance
# matrix plus core distances. modified to be used in Persistable pipeline.
# Originally in scipy/scipy/cluster/_hierarchy.pyx in
# https://github.com/scipy/scipy/commit/2f4bd4234370adcd7085bcc145f9c302f231408f

import numpy as np
cimport numpy as np

from relabel_dendrogram import label


cdef extern from "numpy/npy_math.h":
    cdef enum:
        NPY_INFINITYF

def stepwise_dendrogram_with_core_distances(int n, double[:,:] dists, double[:] core_distances):
    Z_arr = np.empty((n - 1, 3))
    cdef double[:, :] Z = Z_arr

    # Which nodes were already merged.
    cdef int[:] merged = np.zeros(n, dtype=np.intc)

    cdef double[:] D = np.empty(n)
    D[:] = NPY_INFINITYF

    cdef int i, k, x, y = 0
    cdef double dist, current_min

    x = 0
    for k in range(n - 1):
        current_min = NPY_INFINITYF
        merged[x] = 1
        for i in range(n):
            if merged[i] == 1:
                continue

            #dist = max(dists[condensed_index(n, x, i)],core_distances[x],core_distances[i])
            dist = max(dists[x, i],core_distances[x],core_distances[i])
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

    # Find correct cluster labels and compute cluster sizes inplace.
    label(Z_arr, n)

    return Z_arr