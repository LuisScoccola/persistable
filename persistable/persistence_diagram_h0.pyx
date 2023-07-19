# cython: boundscheck=False
# cython: nonecheck=False
# cython: wraparound=False
# cython: initializedcheck=False

# Authors: Luis Scoccola
# License: 3-clause BSD



cimport cython
import numpy as np
cimport numpy as np
ctypedef np.uint8_t uint8

from borrowed._hdbscan_boruvka cimport BoruvkaUnionFind


cpdef persistence_diagram_h0(double end, double[:] heights, np.int64_t[:,:] merges, double[:] merges_heights):
    cdef np.int64_t n_points = heights.shape[0]
    cdef np.int64_t n_merges = merges.shape[0]
    # this orders the point by appearance
    cdef np.int64_t[:] appearances = np.argsort(heights)
    # contains the current clusters
    cdef BoruvkaUnionFind uf = BoruvkaUnionFind(len(heights))
    # contains the birth time of clusters that are alive
    cdef double[:] clusters_birth = np.full(n_points, -1, dtype=float)
    cdef uint8[:] clusters_died = np.zeros(n_points, dtype=np.uint8)
    # contains the persistence diagram
    cdef list pd = []
    # height index
    cdef np.int64_t hind = 0
    # merge index
    cdef np.int64_t mind = 0
    if len(appearances) == 0:
        return []

    cdef double current_appearence_height = heights[appearances[0]]
    cdef double current_merge_height
    cdef np.int64_t[:] xy
    cdef np.int64_t x
    cdef np.int64_t y
    cdef np.int64_t rx
    cdef np.int64_t ry
    cdef double bx
    cdef double by
    cdef double elder_birth
    cdef double younger_birth
    cdef np.int64_t to_delete

    if len(merges_heights) == 0:
        current_merge_height = end
    else:
        current_merge_height = merges_heights[0]
    while True:
        # while there is no merge
        while (
            hind < n_points
            and heights[appearances[hind]] <= current_merge_height
            and heights[appearances[hind]] < end
        ):
            # add all points that are born as new clusters
            clusters_birth[uf.find(appearances[hind])] = heights[appearances[hind]]
            hind += 1
            if hind == n_points:
                current_appearence_height = end
            else:
                current_appearence_height = heights[appearances[hind]]
        # while there is no cluster being born
        while (
            mind < n_merges
            and merges_heights[mind] < current_appearence_height
            and merges_heights[mind] < end
        ):
            xy = merges[mind]
            x, y = xy
            rx = uf.find(x)
            ry = uf.find(y)
            # if they were not already merged
            if rx != ry:
                # if both clusters are alive, merge them and add a bar to the pd
                if not clusters_died[rx] and not clusters_died[ry]:
                    bx = clusters_birth[rx]
                    by = clusters_birth[ry]
                    elder_birth, younger_birth = min(bx, by), max(bx, by)
                    pd.append([younger_birth, merges_heights[mind]])
                    uf.union_(x, y)
                    rxy = uf.find(x)
                    clusters_birth[rxy] = elder_birth
                # if both clusters are already dead, just merge them into a dead cluster
                elif clusters_died[rx] and clusters_died[ry]:
                    uf.union_(x, y)
                    rxy = uf.find(x)
                    clusters_died[rxy] = True
                # if only one of them is dead
                else:
                    # we make it so that ry already died and rx just died
                    if clusters_died[rx]:
                        x, y = y, x
                        rx, ry = ry, rx
                    # merge the clusters into a dead cluster
                    uf.union_(x, y)
                    rxy = uf.find(x)
                    clusters_died[rxy] = True
            mind += 1
            if mind == n_merges:
                current_merge_height = end
            else:
                current_merge_height = merges_heights[mind]
        if (hind == n_points or heights[appearances[hind]] >= end) and (
            mind == n_merges or merges_heights[mind] >= end
        ):
            break
    # go through all clusters that have been born but haven't died 
    for x in range(n_points):
        rx = uf.find(x)
        if (clusters_birth[rx] != -1) and (not clusters_died[rx]):
            pd.append([clusters_birth[rx], end])
            clusters_died[rx] = True
    return pd
