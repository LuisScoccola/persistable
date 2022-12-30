# cython: boundscheck=False
# cython: nonecheck=False
# cython: wraparound=False
# cython: initializedcheck=False



import numpy as np

from .borrowed._hdbscan_boruvka cimport BoruvkaUnionFind


cpdef persistence_diagram_h0(double end, double[:] heights, long[:,:] merges, double[:] merges_heights):
    cdef int n_points = heights.shape[0]
    cdef int n_merges = merges.shape[0]
    # this orders the point by appearance
    cdef long[:] appearances = np.argsort(heights)
    # contains the current clusters
    cdef BoruvkaUnionFind uf = BoruvkaUnionFind(len(heights))
    # contains the birth time of clusters that are alive
    cdef dict clusters_birth = {}
    cdef dict clusters_died = {}
    # contains the persistence diagram
    cdef list pd = []
    cdef double[:,:] pd_array
    # height index
    cdef long hind = 0
    # merge index
    cdef long mind = 0
    if len(appearances) == 0:
        return np.array([])
    cdef double current_appearence_height = heights[appearances[0]]
    cdef double current_merge_height
    cdef long[:] xy
    cdef long x
    cdef long y
    cdef long rx
    cdef long ry
    cdef double bx
    cdef double by
    cdef double elder_birth
    cdef double younger_birth
    cdef long to_delete
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
            ##uf.add(appearances[hind])
            ##clusters_birth[appearances[hind]] = heights[appearances[hind]]
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
                if rx not in clusters_died and ry not in clusters_died:
                    bx = clusters_birth[rx]
                    by = clusters_birth[ry]
                    elder_birth, younger_birth = min(bx, by), max(bx, by)
                    pd.append([younger_birth, merges_heights[mind]])
                    #del clusters_birth[rx]
                    #del clusters_birth[ry]
                    uf.union_(x, y)
                    rxy = uf.find(x)
                    clusters_birth[rxy] = elder_birth
                # if both clusters are already dead, just merge them into a dead cluster
                elif rx in clusters_died and ry in clusters_died:
                    uf.union_(x, y)
                    rxy = uf.find(x)
                    clusters_died[rxy] = True
                # if only one of them is dead
                else:
                    # we make it so that ry already died and rx just died
                    if rx in clusters_died:
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
        ##if x in uf._indices:
        rx = uf.find(x)
        if (rx in clusters_birth) and (rx not in clusters_died):
            pd.append([clusters_birth[rx], end])
            clusters_died[rx] = True
    return pd
