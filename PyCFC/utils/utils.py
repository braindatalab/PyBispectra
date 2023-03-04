"""Helper tools for processing CFC results."""

import numpy as np
from numba import njit


@njit
def fast_find_first(vector: np.ndarray, value: float) -> int:
    """Quickly find the first index of a value in a 1D array using Numba.

    PARAMETERS
    ----------
    vector : NumPy ndarray
    -   1D array to find `value` in.

    value : float
    -   value to find in `vector`.

    RETURNS
    -------
    index : int
    -   First index of `value` in `vector`.

    NOTES
    -----
    -   Does not check if `vector` is a 1D NumPy array or if `value` is a
        single value for speed.
    """
    for idx, val in enumerate(vector):
        if val == value:
            return idx
    raise ValueError("`value` is not present in `vector`.")
