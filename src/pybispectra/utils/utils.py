"""Helper tools for processing results."""

from warnings import warn

from mne import Info, create_info
from numba import njit
import numpy as np
from pqdm.processes import pqdm
import scipy as sp


def compute_fft(
    data: np.ndarray,
    sfreq: int,
    n_jobs: int = 1,
    verbose: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the FFT on real-valued data.

    As the data is assumed to be real-valued, only those values corresponding
    to the positive frequencies are returned.

    Parameters
    ----------
    data : numpy.ndarray of float, shape of [epochs x channels x times]
        Real-valued data to compute the FFT on.

    sfreq : int
        Sampling frequency of the data in Hz.

    n_jobs : int (default ``1``)
        Number of jobs to run in parallel.

    verbose : bool (default True)
        Whether or not to report the status of the processing.

    Returns
    -------
    fft : numpy.ndarray of float, shape of [epochs x channels x frequencies]
        FFT coefficients for the positive frequencies of ``data``.

    freqs : numpy.ndarray of float, shape of [frequencies]
        Frequencies (in Hz) in ``fft``.
    """
    if not isinstance(data, np.ndarray):
        raise TypeError("`data` must be a NumPy NDArray.")
    if data.ndim != 3:
        raise ValueError("`data` must be a 3D array.")

    if not isinstance(n_jobs, int):
        raise TypeError("`n_jobs` must be an integer.")
    if n_jobs < 1:
        raise ValueError("`n_jobs` must be >= 1.")

    if not isinstance(sfreq, int):
        if isinstance(sfreq, float):
            if verbose:
                warn(
                    "`sfreq` is a float. Converting it to an int.", UserWarning
                )
        else:
            raise TypeError("`sfreq` must be an int.")

    if verbose and not np.isreal(data).all():
        warn("`data` is expected to be real-valued.", UserWarning)

    if verbose:
        print("Computing FFT on the data...")

    freqs = np.linspace(0.0, sfreq / 2.0, int(sfreq) + 1)

    window = np.hanning(data.shape[2])

    args = [
        {"a": sp.signal.detrend(chan_data) * window}
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


@njit
def fast_find_first(vector: np.ndarray, value: float | int) -> int:
    """Quickly find the first index of a value in a 1D array using Numba.

    Parameters
    ----------
    vector : numpy.ndarray of float or int
        1D array to find ``value`` in.

    value : float | int
        value to find in ``vector``.

    Returns
    -------
    index : int
        First index of ``value`` in ``vector``.

    Notes
    -----
    Does not perform checks in inputs for speed.
    """
    for idx, val in enumerate(vector):
        if val == value:
            return idx
    raise ValueError("`value` is not present in `vector`.")


def compute_rank(data: np.ndarray, sv_tol: float = 1e-5) -> int:
    """Compute the min. rank of data over epochs from non-zero singular values.

    Parameters
    ----------
    data : numpy.ndarray, shape of [epochs x channels x times]
        Data to find the rank of.

    sv_tol : float (default ``1e-5``)
        Tolerance to use to define non-zero singular values based on the
        largest singular value.

    Returns
    -------
    rank : int
        Minimum rank of ``data`` over epochs.
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
