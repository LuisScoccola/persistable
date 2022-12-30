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

    #def __init__(self, size):
    #    self._parent_arr = np.arange(size, dtype=np.intp)
    #    self._parent = (<np.intp_t[:size:1]> (<np.intp_t *>
    #                                          self._parent_arr.data))
    #    self._rank_arr = np.zeros(size, dtype=np.uint8)
    #    self._rank = (<np.uint8_t[:size:1]> (<np.uint8_t *>
    #                                         self._rank_arr.data))
    #    self.is_component = np.ones(size, dtype=bool)

    cpdef int union_(self, np.intp_t x, np.intp_t y) except -1

    cpdef np.intp_t find(self, np.intp_t x) except -1

    cpdef np.ndarray[np.intp_t, ndim=1] components(self)
