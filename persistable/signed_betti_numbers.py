import numpy as np
import itertools


def signed_betti(hilbert_function):
    # number of dimensions
    n = len(hilbert_function.shape)
    # pad with zeros at the end so np.roll does not roll over
    hf_padded = np.pad(hilbert_function,[[0,1]]*n)
    # all relevant shifts (e.g., if n=2, (0,0), (0,1), (1,0), (1,1))
    shifts = np.array(list(itertools.product([0,1],repeat=n)), dtype=int)
    bn = np.zeros(hf_padded.shape, dtype=int)
    for shift in shifts:
        bn += ((-1)**np.sum(shift)) * np.roll(hf_padded,shift,axis=range(n))
    # remove the padding
    slices = np.ix_( *[range(0,hilbert_function.shape[i]) for i in range(n)] )
    return bn[slices]


