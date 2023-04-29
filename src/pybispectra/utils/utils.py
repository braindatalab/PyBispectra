"""Helper tools for processing results."""

from typing import Callable
from warnings import warn

from mne import Info, create_info
from numba import njit
import numpy as np
from pqdm.processes import pqdm
import scipy as sp


def compute_fft(
    data: np.ndarray,
    sfreq: int | float,
    n_points: int | None = None,
    window: str = "hanning",
    return_neg_freqs: bool = False,
    n_jobs: int = 1,
    verbose: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the FFT on real-valued data.

    As the data is assumed to be real-valued, only those values corresponding
    to the positive frequencies are returned by default (see
    :param:`return_neg_freqs`).

    Parameters
    ----------
    data : numpy.ndarray of float, shape of [epochs, channels, times]
        Real-valued data to compute the FFT on.

    sfreq : int | float
        Sampling frequency of the data in Hz.

    n_points : int | None (default ``None``)
        Number of points in the FFT. If ``None``, is equal to the number of
        times + 1.

    window : str (default ``"hanning"``)
        Type of window to apply to :param:`data` before computing the FFT.
        Accepts ``"hanning"`` and ``"hamming"``. See :func:`numpy.hanning` and
        :func:`numpy.hamming`.

    return_neg_freqs : bool (default ``False``)
        Whether or not to return the FFT coefficients for negative frequencies.

    n_jobs : int (default ``1``)
        Number of jobs to run in parallel.

    verbose : bool (default ``True``)
        Whether or not to report the status of the processing.

    Returns
    -------
    fft : numpy.ndarray of float, shape of [epochs, channels, frequencies]
        FFT coefficients of :param:`data`.

    freqs : numpy.ndarray of float, shape of [frequencies]
        Frequencies (in Hz) in :param:`fft`.
    """
    n_points, window_func = _compute_fft_input_checks(
        data, sfreq, n_points, window, return_neg_freqs, n_jobs, verbose
    )

    if verbose:
        print("Computing FFT on the data...")

    freqs = np.fft.fftfreq(n=n_points, d=1 / sfreq)
    if not return_neg_freqs:
        freqs = np.abs(freqs)
        freqs = freqs[: freqs.argmax()]

    window = window_func(data.shape[2])

    args = [
        {"a": sp.signal.detrend(chan_data) * window, "n": n_points}
        for chan_data in data.transpose(1, 0, 2)
    ]

    fft = np.array(
        pqdm(
            args,
            np.fft.fft,
            n_jobs,
            argument_type="kwargs",
            desc="Processing channels...",
            disable=not verbose,
        )
    ).transpose(1, 0, 2)

    if verbose:
        print("    [FFT computation finished]\n")

    return fft[..., : len(freqs)], freqs


def _compute_fft_input_checks(
    data: np.ndarray,
    sfreq: int | float,
    n_points: int | None,
    window: str,
    return_neg_freqs: bool,
    n_jobs: int,
    verbose: bool,
) -> tuple[int, Callable]:
    """Check inputs for computing FFT.

    Returns
    -------
    n_points : int

    window_func : Callable
    """
    if not isinstance(data, np.ndarray):
        raise TypeError("`data` must be a NumPy array.")
    if data.ndim != 3:
        raise ValueError("`data` must be a 3D array.")

    if n_points is None:
        n_points = data.shape[2] + 1
    elif not isinstance(n_points, int):
        raise TypeError("`n_points` must be an integer")

    if not isinstance(n_jobs, int):
        raise TypeError("`n_jobs` must be an integer.")
    if n_jobs < 1:
        raise ValueError("`n_jobs` must be >= 1.")

    if not isinstance(sfreq, int) and not isinstance(sfreq, float):
        raise TypeError("`sfreq` must be an int or a float.")

    if not isinstance(window, str):
        raise TypeError("`window` must be a str.")
    if window not in ["hanning", "hamming"]:
        raise ValueError("The requested `window` type is not recognised.")
    if window == "hanning":
        window_func = np.hanning
    else:
        window_func = np.hamming

    if not isinstance(return_neg_freqs, bool):
        raise TypeError("`return_neg_freqs` must be a bool.")

    if verbose and not np.isreal(data).all():
        warn("`data` is expected to be real-valued.", UserWarning)

    return n_points, window_func


@njit
def fast_find_first(vector: np.ndarray, value: int | float) -> int:
    """Quickly find the first index of a value in a 1D array using Numba.

    Parameters
    ----------
    vector : numpy.ndarray of int or float
        1D array to find :param:`value` in.

    value : int | float
        Value to find in :param:`vector`.

    Returns
    -------
    index : int
        First index of :param:`value` in :param:`vector`.

    Notes
    -----
    Does not perform checks on inputs for speed.
    """
    for idx, val in enumerate(vector):
        if val == value:
            return idx
    raise ValueError("`value` is not present in `vector`.")


def compute_rank(data: np.ndarray, sv_tol: int | float = 1e-5) -> int:
    """Compute the min. rank of data over epochs from non-zero singular values.

    Parameters
    ----------
    data : numpy.ndarray, shape of [epochs, channels, times]
        Data to find the rank of.

    sv_tol : int | float (default ``1e-5``)
        Tolerance to use to define non-zero singular values, based on the
        largest singular value.

    Returns
    -------
    rank : int
        Minimum rank of :param:`data` over epochs.
    """
    if not isinstance(data, np.ndarray):
        raise TypeError("`data` must be a NumPy array.")
    if data.ndim != 3:
        raise ValueError("`data` must be a 3D array.")

    if not isinstance(sv_tol, float) and not isinstance(sv_tol, int):
        raise TypeError("`sv_tol` must be a float or an int.")

    singular_vals = np.linalg.svd(data, compute_uv=False).min(axis=0)

    return np.count_nonzero(singular_vals > singular_vals[0] * sv_tol)


def _create_mne_info(n_chans: int, sfreq: float) -> Info:
    """Create an MNE Info object.

    Parameters
    ----------
    n_chans : int
        Number of channels in the data to create names and types for.

    sfreq : float
        Sampling frequency of the data (in Hz).

    Returns
    -------
    info : MNE Info
        MNE Info object.

    Notes
    -----
    Names are set as ``[str(i) for i in range(n_chans)]``, and channel types
    are all set to EEG (any MNE *data* type could be used; note that not all
    MNE channel types are recognised as data types).
    """
    ch_names = [str(i) for i in range(n_chans)]
    ch_types = ["eeg" for _ in range(n_chans)]

    return create_info(ch_names, sfreq, ch_types, verbose=False)


def _generate_data(
    n_epochs: int, n_chans: int, n_times: int, seed: int = 44
) -> np.ndarray:
    """Generate random data of the specified shape."""
    random = np.random.RandomState(seed)
    return random.rand(n_epochs, n_chans, n_times)
