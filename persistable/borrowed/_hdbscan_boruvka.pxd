#!python
# cython: boundscheck=False
# cython: nonecheck=False
# cython: wraparound=False
# cython: initializedcheck=False

import cython
cimport cython

import numpy as np
cimport numpy as np


cdef class BoruvkaUnionFind (object):
    cdef np.ndarray _parent_arr
    cdef np.intp_t[::1] _parent
    cdef np.ndarray _rank_arr
    cdef np.uint8_t[::1] _rank
    cdef np.ndarray is_component

    cpdef int union_(self, np.intp_t x, np.intp_t y) except -1

    cpdef np.intp_t find(self, np.intp_t x) except -1

    cpdef np.ndarray[np.intp_t, ndim=1] components(self)
