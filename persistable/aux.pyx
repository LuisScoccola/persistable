# cython: boundscheck=False, wraparound=False, cdivision=True

# Authors: Luis Scoccola
# License: 3-clause BSD

import numpy as np
cimport numpy as np
np.import_array()

DTYPE = np.float64
ctypedef np.float64_t DTYPE_t


def lazy_intersection(np.ndarray[DTYPE_t, ndim=1] increasing, np.ndarray[DTYPE_t, ndim=1] increasing2, np.float64_t s0, np.float64_t k0) :
    # find first occurence of s0 - (s0/k0) * increasing[i]) <= increasing2[i]
    assert increasing.dtype == DTYPE and increasing2.dtype == DTYPE
    cdef np.float64_t mu = s0/k0
    cdef int first = 0
    cdef int last = increasing.shape[0]-1
    cdef int midpoint
    cdef int res1
    cdef int res2
    with nogil:
        if s0 - mu * increasing[first] <= increasing2[first] :
            res1, res2 = first, False
        elif s0 - mu * increasing[last] > increasing2[last] :
            res1, res2 = last, True
        else:
            while first+1 < last :
                midpoint = (first + last)//2
                if s0 - mu * increasing[midpoint] <= increasing2[midpoint] :
                    last = midpoint
                else:
                    first = midpoint
            res1, res2 = last, False
    return res1, res2
