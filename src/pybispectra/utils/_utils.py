"""Private helper tools for processing results."""

import numpy as np
from mne import Info, create_info
from mne.parallel import parallel_func
from mne.utils import ProgressBar, set_log_level
from numba import njit

from pybispectra.utils._defaults import _precision


# Aliases for type checking
_int_like = (int, np.integer)
_float_like = (float, np.floating)
_number_like = _int_like + _float_like


def _compute_in_parallel(
    func: callable,
    loop_kwargs: list[dict],
    static_kwargs: dict,
    output: np.ndarray,
    message: str,
    n_jobs: int,
    verbose: bool,
    prefer: str = "processes",
) -> np.ndarray:
    """Parallelise a function with a progress bar.

    Parameters
    ----------
    func : callable
        Function to parallelise.

    loop_kwargs : list of dict
        List of keyword arguments to pass to the function that change for each iteration
        of the parallelisation.

    static_kwargs : dict
        Dictionary of keyword arguments to pass to the function that do not change
        across iterations.

    output : numpy.ndarray
        Array to store the output of the computation. Values for each iteration of the
        parallelisation are stored in the first dimension, which must be at least as
        large as the length of the values in ``loop_kwargs``.

    message : str
        Message to display in the progress bar.

    n_jobs : int
        Number of jobs to run in parallel.

    verbose : bool
        Whether or not to report the progress of the processing.

    prefer : str (default "processes")
        Whether to use "threads" or "processes" for parallelisation.

    Returns
    -------
    output : numpy.ndarray
        Array with the output of the computation.

    Notes
    -----
    Relies on the MNE progress bar and parallel implementations. Does not perform checks
    on inputs for speed.
    """
    n_steps = len(loop_kwargs)
    n_blocks = int(np.ceil(n_steps / n_jobs))
    parallel, my_parallel_func, _ = parallel_func(
        func, n_jobs, prefer=prefer, verbose=verbose
    )
    old_log_level = set_log_level(
        verbose="INFO" if verbose else "WARNING", return_old_level=True
    )  # need to set log level that is passed to tqdm
    for block_i in ProgressBar(range(n_blocks), mesg=message):
        idcs = _get_block_indices(block_i, n_steps, n_jobs)
        output[idcs] = parallel(
            my_parallel_func(**loop_kwargs[idx], **static_kwargs) for idx in idcs
        )
    set_log_level(verbose=old_log_level)  # reset log level

    return output


def _get_block_indices(block_i: int, limit: int, n_jobs: int) -> np.ndarray:
    """Get the indices for a block of parallel computation, capped by a limit.

    Parameters
    ----------
    block_i : int
        Index of the block to get indices for.

    limit : int
        Maximum index to return.

    n_jobs : int
        Number of jobs to run in parallel.

    Returns
    -------
    indices : numpy.ndarray of int
        Indices for the block of parallel computation.
    """
    return np.arange(block_i * n_jobs, np.min([(block_i + 1) * n_jobs, limit]))


@njit
def _fast_find_first(
    vector: np.ndarray, value: int | float, start_idx: int = 0
) -> int:  # pragma: no cover
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
def _compute_pearsonr_2d(
    x: np.ndarray, y: np.ndarray, precision: type
) -> np.ndarray:  # pragma: no cover
    """Compute Pearson correlation for epochs over time.

    Parameters
    ----------
    x : numpy.ndarray, shape of [epochs, times]
        Array of time-series values to compute correlation of with ``y``.

    y : numpy.ndarray, shape of [epochs, times]
        Array of time-series values to compute correlation of with ``x``.

    precision : type
        Precision to use for the computation. Either ``numpy.float32`` (single) or
        ``numpy.float64`` (double).

    Returns
    -------
    pearsonr : numpy.ndarray, shape of [epochs]
        Correlation coefficient between ``x`` and ``y`` over time for each epoch.

    Notes
    -----
    Does not perform checks on inputs for speed.
    """
    x_minus_mean = np.full(x.shape, fill_value=np.nan, dtype=precision)
    y_minus_mean = np.full(y.shape, fill_value=np.nan, dtype=precision)
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

    return np.divide(numerator, denominator).astype(precision)


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
    Names are set as ``[str(i) for i in range(n_chans)]``, and channel types are all set
    to EEG (any MNE *data* type could be used; note that not all MNE channel types are
    recognised as data types).
    """
    ch_names = [str(i) for i in range(n_chans)]
    ch_types = ["eeg" for _ in range(n_chans)]  # must be an MNE data ch. type

    return create_info(ch_names, sampling_freq, ch_types, verbose=False)


def _generate_data(
    n_epochs: int, n_chans: int, n_times: int, seed: int = 44
) -> np.ndarray:
    """Generate random data of the specified shape."""
    random = np.random.RandomState(seed)
    return random.rand(n_epochs, n_chans, n_times).astype(_precision.real)
