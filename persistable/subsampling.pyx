# cython: boundscheck=False
# cython: nonecheck=False

# Authors: Luis Scoccola
# License: 3-clause BSD

import numpy as np
cimport numpy as np

from libc.float cimport DBL_MAX

from borrowed.dist_metrics cimport DistanceMetric


#cpdef np.ndarray[np.int64_t, ndim=1] close_subsample_distance_matrix(subsample_size, seed=0):


cpdef np.ndarray[np.int64_t, ndim=1] close_subsample_fast_metric(
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
    cdef np.double_t[:] radii = np.zeros(subsample_size)

    cdef np.double_t[:] ds = np.full(dim, np.inf)

    idx = random_start
    idx_perm_view[0] = idx
    for j in range(0, dim):
        val1 = ds[j]
        val2 = dist_metric.dist(&raw_data_ptr[num_features * idx],
                                 &raw_data_ptr[num_features * j],
                                 num_features)
        ds[j] = min(val1, val2)

    for i in range(1, subsample_size):
        idx = np.argmax(ds)
        idx_perm_view[i] = idx
        radii[i - 1] = ds[idx]
        for j in range(0, dim):
            val1 = ds[j]
            val2 = dist_metric.dist(&raw_data_ptr[num_features * idx],
                                     &raw_data_ptr[num_features * j],
                                     num_features)
            ds[j] = min(val1, val2)

    radii[-1] = np.max(ds)

    return idx_perm
    #return idx_perm, radii

#        #for i in range(1, dim):
#        #    in_tree[current_node] = 1
#    
#        #    current_node_core_distance = core_distance_ptr[current_node]
#    
#        #    new_distance = DBL_MAX
#        #    source_node = 0
#        #    new_node = 0
#    
#        #    for j in range(dim):
#        #        if in_tree[j]:
#        #            continue
#    
#        #        right_value = current_distances[j]
#        #        right_source = current_sources[j]
#    
#        #        left_value = dist_metric.dist(&raw_data_ptr[num_features *
#        #                                                    current_node],
#        #                                      &raw_data_ptr[num_features * j],
#        #                                      num_features)
#        #        left_source = current_node
#    
#        #        core_value = core_distance_ptr[j]
#        #        if (current_node_core_distance > right_value or
#        #                core_value > right_value or
#        #                left_value > right_value):
#        #            if right_value < new_distance:
#        #                new_distance = right_value
#        #                source_node = right_source
#        #                new_node = j
#        #            continue
#    
#        #        if core_value > current_node_core_distance:
#        #            if core_value > left_value:
#        #                left_value = core_value
#        #        else:
#        #            if current_node_core_distance > left_value:
#        #                left_value = current_node_core_distance
#    
#        #        if left_value < right_value:
#        #            current_distances[j] = left_value
#        #            current_sources[j] = left_source
#        #            if left_value < new_distance:
#        #                new_distance = left_value
#        #                source_node = left_source
#        #                new_node = j
#        #        else:
#        #            if right_value < new_distance:
#        #                new_distance = right_value
#        #                source_node = right_source
#        #                new_node = j
#    
#        #    result[i - 1, 0] = <double> source_node
#        #    result[i - 1, 1] = <double> new_node
#        #    result[i - 1, 2] = new_distance
#        #    current_node = new_node
#    
#    order = np.argsort(result_arr[:,2], kind='mergesort')
#    result_arr = result_arr[order]
#    # do not relabel since we don't really need it, and this makes things easier
#    # when combining different hierarchical clusterings
#    #label(result_arr, dim, dim)
#    return result_arr
