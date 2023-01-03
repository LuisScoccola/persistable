# cython: boundscheck=False
# cython: nonecheck=False
# Minimum spanning tree single linkage implementation for hdbscan
# Authors: Leland McInnes, Steve Astels
# License: 3-clause BSD
# Modified to work with the Persistable pipeline
# taken from: https://github.com/scikit-learn-contrib/hdbscan/commit/fafec0c1611aaa12d7587f829d45d3c83f6057f4

import numpy as np
cimport numpy as np

from libc.float cimport DBL_MAX

from dist_metrics cimport DistanceMetric


cpdef np.ndarray[np.double_t, ndim=2] mst_linkage_core_vector(
        np.ndarray[np.double_t, ndim=2, mode='c'] raw_data,
        np.ndarray[np.double_t, ndim=1, mode='c'] core_distance,
        DistanceMetric dist_metric):

    cdef np.ndarray result_arr 
    cdef np.ndarray core_distance_arr
    cdef np.double_t *core_distance_ptr

    cdef np.intp_t dim = raw_data.shape[0]
    result_arr = np.zeros((dim - 1, 3))

    core_distance_arr = np.asarray(core_distance,dtype=np.double)
    cdef np.double_t[::1] core_distance_ = (<np.double_t[:dim:1]> (
        <np.double_t *> core_distance_arr.data))

    core_distance_ptr = <np.double_t *> &core_distance_[0]

    cdef np.intp_t num_features
    num_features = raw_data.shape[1]

    cdef np.ndarray[np.double_t, ndim=1] current_distances_arr
    cdef np.ndarray[np.intp_t, ndim=1] current_sources_arr
    cdef np.ndarray[np.int8_t, ndim=1] in_tree_arr
    
    cdef np.double_t * current_distances
    cdef np.intp_t * current_sources
    cdef np.double_t * raw_data_ptr
    cdef np.int8_t * in_tree
    cdef np.double_t[:, ::1] raw_data_view
    cdef np.double_t[:, ::1] result
    
    cdef np.ndarray label_filter
    
    cdef np.intp_t current_node
    cdef np.intp_t source_node
    cdef np.intp_t right_node
    cdef np.intp_t left_node
    cdef np.intp_t new_node
    cdef np.intp_t i
    cdef np.intp_t j
    
    cdef np.double_t current_node_core_distance
    cdef np.double_t right_value
    cdef np.double_t left_value
    cdef np.double_t core_value
    cdef np.double_t new_distance
    cdef np.intp_t right_source
    cdef np.intp_t left_source

    in_tree_arr = np.zeros(dim, dtype=np.int8)
    current_node = 0
    current_distances_arr = np.infty * np.ones(dim)
    current_sources_arr = np.ones(dim, dtype=np.intp)
    
    result = (<np.double_t[:dim - 1, :3:1]> (<np.double_t *> result_arr.data))
    in_tree = (<np.int8_t *> in_tree_arr.data)
    current_distances = (<np.double_t *> current_distances_arr.data)
    current_sources = (<np.intp_t*> current_sources_arr.data)

    raw_data_view = (<np.double_t[:raw_data.shape[0], :raw_data.shape[1]:1]> (
        <np.double_t *> raw_data.data))
    raw_data_ptr = (<np.double_t *> &raw_data_view[0, 0])

    with nogil:
        for i in range(1, dim):
    
            in_tree[current_node] = 1
    
            current_node_core_distance = core_distance_ptr[current_node]
    
            new_distance = DBL_MAX
            source_node = 0
            new_node = 0
    
            for j in range(dim):
                if in_tree[j]:
                    continue
    
                right_value = current_distances[j]
                right_source = current_sources[j]
    
                left_value = dist_metric.dist(&raw_data_ptr[num_features *
                                                            current_node],
                                              &raw_data_ptr[num_features * j],
                                              num_features)
                left_source = current_node
    
                core_value = core_distance_ptr[j]
                if (current_node_core_distance > right_value or
                        core_value > right_value or
                        left_value > right_value):
                    if right_value < new_distance:
                        new_distance = right_value
                        source_node = right_source
                        new_node = j
                    continue
    
                if core_value > current_node_core_distance:
                    if core_value > left_value:
                        left_value = core_value
                else:
                    if current_node_core_distance > left_value:
                        left_value = current_node_core_distance
    
                if left_value < right_value:
                    current_distances[j] = left_value
                    current_sources[j] = left_source
                    if left_value < new_distance:
                        new_distance = left_value
                        source_node = left_source
                        new_node = j
                else:
                    if right_value < new_distance:
                        new_distance = right_value
                        source_node = right_source
                        new_node = j
    
            result[i - 1, 0] = <double> source_node
            result[i - 1, 1] = <double> new_node
            result[i - 1, 2] = new_distance
            current_node = new_node
    
    order = np.argsort(result_arr[:,2], kind='mergesort')
    result_arr = result_arr[order]
    # do not relabel since we don't really need it, and this makes things easier
    # when combining different hierarchical clusterings
    #label(result_arr, dim, dim)
    return result_arr
