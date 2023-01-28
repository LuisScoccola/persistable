# cython: boundscheck=False
# cython: nonecheck=False

# Authors: Luis Scoccola
# License: 3-clause BSD

import numpy as np
cimport numpy as np

from libc.float cimport DBL_MAX

from borrowed.dist_metrics cimport DistanceMetric


cpdef close_subsample_distance_matrix(
    np.int64_t subsample_size,
    np.ndarray[np.double_t, ndim=2, mode='c'] dist_mat,
    random_start=0
):
    cdef np.int64_t dim = dist_mat.shape[0]

    cdef np.double_t[:, ::1] dist_mat_view = dist_mat

    cdef np.double_t val1
    cdef np.double_t val2

    cdef np.int64_t idx

    idx_perm = np.zeros(subsample_size, dtype=np.int64)
    cdef np.int64_t[:] idx_perm_view = idx_perm
    radii = np.zeros(subsample_size)
    cdef np.double_t[:] radii_view = radii
    representatives = np.zeros(dim, dtype=np.int64)
    cdef np.int64_t[:] representatives_view = representatives

    cdef np.double_t[:] ds = np.full(dim, np.inf)

    idx = random_start
    idx_perm_view[0] = idx
    for j in range(0, dim):
        ds[j] = dist_mat[idx,j]
        representatives_view[j] = 0

    for i in range(1, subsample_size):
        idx = np.argmax(ds)
        idx_perm_view[i] = idx
        radii_view[i - 1] = ds[idx]
        for j in range(0, dim):
            val1 = ds[j]
            val2 = dist_mat[idx,j]
            if val1 < val2:
                ds[j] = val1
            else:
                ds[j] = val2
                representatives_view[j] = i

    radii_view[-1] = np.max(ds)

    return idx_perm, representatives
    #return idx_perm, radii, representatives


cpdef close_subsample_fast_metric(
    np.int64_t subsample_size,
    np.ndarray[np.double_t, ndim=2, mode='c'] raw_data,
    DistanceMetric dist_metric,
    random_start=0
):
    cdef np.int64_t dim = raw_data.shape[0]
    cdef np.int64_t num_features
    num_features = raw_data.shape[1]

    cdef np.double_t[:, ::1] raw_data_view

    cdef np.double_t val1
    cdef np.double_t val2

    cdef np.int64_t idx

    raw_data_view = (<np.double_t[:raw_data.shape[0], :raw_data.shape[1]:1]> (
        <np.double_t *> raw_data.data))
    raw_data_ptr = (<np.double_t *> &raw_data_view[0, 0])

    idx_perm = np.zeros(subsample_size, dtype=np.int64)
    cdef np.int64_t[:] idx_perm_view = idx_perm
    radii = np.zeros(subsample_size)
    cdef np.double_t[:] radii_view = radii
    representatives = np.zeros(dim, dtype=np.int64)
    cdef np.int64_t[:] representatives_view = representatives

    cdef np.double_t[:] ds = np.full(dim, np.inf)

    idx = random_start
    idx_perm_view[0] = idx

    for j in range(0, dim):
        ds[j] = dist_metric.dist(&raw_data_ptr[num_features * idx],
                                 &raw_data_ptr[num_features * j],
                                 num_features)
        representatives_view[j] = 0

    for i in range(1, subsample_size):
        idx = np.argmax(ds)
        idx_perm_view[i] = idx
        radii_view[i - 1] = ds[idx]
        for j in range(0, dim):
            val1 = ds[j]
            val2 = dist_metric.dist(&raw_data_ptr[num_features * idx],
                                     &raw_data_ptr[num_features * j],
                                     num_features)
            if val1 < val2:
                ds[j] = val1
            else:
                ds[j] = val2
                representatives_view[j] = i

    radii_view[-1] = np.max(ds)

    return idx_perm, representatives
    #return idx_perm, radii, representatives
