# cython: boundscheck=False, wraparound=False, cdivision=True

# Copyright (c) 2001-2002 Enthought, Inc. 2003-2022, SciPy Developers.
#
# Functions to relabel a dendrogram according to scipy's standard,
# modified to be used in Persistable pipeline. Originally in 
# scipy/scipy/cluster/_hierarchy.pyx in
# https://github.com/scipy/scipy/commit/2f4bd4234370adcd7085bcc145f9c302f231408f


import numpy as np
cimport numpy as np


cdef class LinkageUnionFind:
    """Structure for fast cluster labeling in unsorted dendrogram."""
    cdef int[:] parent
    cdef int[:] size
    cdef int next_label

    def __init__(self, int num_points, int num_merges):
        n = num_points + num_merges
        self.parent = np.arange(n, dtype=np.intc)
        self.next_label = num_points
        self.size = np.ones(n, dtype=np.intc)

    def merge(self, int x, int y):
        self.parent[x] = self.next_label
        self.parent[y] = self.next_label
        cdef int size = self.size[x] + self.size[y]
        self.size[self.next_label] = size
        self.next_label += 1
        return size

    def find(self, int x):
        cdef int p = x

        while self.parent[x] != x:
            x = self.parent[x]

        while self.parent[p] != x:
            p, self.parent[p] = self.parent[p], x

        return x

def label(long[:, :] Z, int num_points, int num_merges):
    """Correctly label clusters in unsorted dendrogram."""
    cdef LinkageUnionFind uf = LinkageUnionFind(num_points, num_merges)
    cdef int i, x, y, x_root, y_root
    for i in range(num_merges):
        x, y = int(Z[i, 0]), int(Z[i, 1])
        x_root, y_root = uf.find(x), uf.find(y)
        if x_root < y_root:
            Z[i, 0], Z[i, 1] = x_root, y_root
        else:
            Z[i, 0], Z[i, 1] = y_root, x_root
        _ = uf.merge(x_root, y_root)
