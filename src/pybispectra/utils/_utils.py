"""Private helper tools for processing results."""

from mne import Info, create_info
from numba import njit
import numpy as np


@njit
def _fast_find_first(
    vector: np.ndarray, value: int | float, start_idx: int = 0
) -> int:
    """Quickly find the first index of a value in a 1D array using Numba.

    Parameters
    ----------
    vector : numpy.ndarray of int or float
        1D array to find ``value`` in.

    value : int | float
        Value to find in ``vector``.

    start_idx : int (default 0)
        Index to start searching for ``value`` in ``vector``.

    Returns
    -------
    index : int
        First index of ``value`` in ``vector``.

    Notes
    -----
    Does not perform checks on inputs for speed.
    """
    for idx, val in enumerate(vector[start_idx:]):
        if val == value:
            return idx + start_idx
    raise ValueError("`value` is not present in `vector`.")


@njit
def _compute_pearsonr_2d(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Compute Pearson correlation for epochs over time.

    Parameters
    ----------
    x : numpy.ndarray, shape of [epochs, times]
        Array of time-series values to compute correlation of with ``y``.

    y : numpy.ndarray, shape of [epochs, times]
        Array of time-series values to compute correlation of with ``x``.

    Returns
    -------
    pearsonr : numpy.ndarray, shape of [epochs]
        Correlation coefficient between ``x`` and ``y`` over time for each
        epoch.

    Notes
    -----
    Does not perform checks on inputs for speed.
    """
    x_minus_mean = np.full(x.shape, fill_value=np.nan, dtype=np.float64)
    y_minus_mean = np.full(y.shape, fill_value=np.nan, dtype=np.float64)
    for idx in range(x.shape[0]):  # same as y.shape[0]
        x_minus_mean[idx] = x[idx] - np.mean(x[idx])
        y_minus_mean[idx] = y[idx] - np.mean(y[idx])

    numerator = np.sum(np.multiply(x_minus_mean, y_minus_mean), axis=-1)
    denominator = np.sqrt(
        np.multiply(
            np.sum(np.square(x_minus_mean), axis=-1),
            np.sum(np.square(y_minus_mean), axis=-1),
        )
    )

    return np.divide(numerator, denominator)


def _create_mne_info(n_chans: int, sampling_freq: float) -> Info:
    """Create an MNE Info object.

    Parameters
    ----------
    n_chans : int
        Number of channels in the data to create names and types for.

    sampling_freq : float
        Sampling frequency of the data (in Hz).

    Returns
    -------
    info : mne.Info
        MNE Info object.

    Notes
    -----
    Names are set as ``[str(i) for i in range(n_chans)]``, and channel types
    are all set to EEG (any MNE *data* type could be used; note that not all
    MNE channel types are recognised as data types).
    """
    ch_names = [str(i) for i in range(n_chans)]
    ch_types = ["eeg" for _ in range(n_chans)]  # must be an MNE data ch. type

    return create_info(ch_names, sampling_freq, ch_types, verbose=False)


def _generate_data(
    n_epochs: int, n_chans: int, n_times: int, seed: int = 44
) -> np.ndarray:
    """Generate random data of the specified shape."""
    random = np.random.RandomState(seed)
    return random.rand(n_epochs, n_chans, n_times)
