"""Public tools for handling data and processing results."""

from multiprocessing import cpu_count
from typing import Callable
from warnings import warn

import pooch
import numpy as np
import scipy as sp
from mne import time_frequency

from pybispectra import __version__
from pybispectra.utils._defaults import _precision
from pybispectra.utils._utils import _compute_in_parallel


def compute_fft(
    data: np.ndarray,
    sampling_freq: int | float,
    n_points: int | None = None,
    window: str = "hanning",
    n_jobs: int = 1,
    verbose: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the fast Fourier transform (FFT) on real-valued data.

    As the data is assumed to be real-valued, only those values corresponding to the
    positive frequencies are returned.

    Parameters
    ----------
    data : ~numpy.ndarray, shape of [epochs, channels, times]
        Real-valued data to compute the FFT on.

    sampling_freq : int | float
        Sampling frequency (in Hz) of ``data``.

    n_points : int | None (default None)
        Number of points to use in the FFT. If :obj:`None`, is equal to the number of
        timepoints in ``data``. For time delay estimation, it is recommended that
        ``n_points=2 * n_times + 1``, where ``n_times`` is the number of timepoints in
        each epoch of ``data``.

    window : ``"hanning"`` | ``"hamming"`` (default ``"hanning"``)
        Type of window to apply to ``data`` before computing the FFT. See
        :func:`numpy.hanning` and :func:`numpy.hamming`.

    n_jobs : int (default ``1``)
        Number of jobs to run in parallel. If ``-1``, all available CPUs are used.

    verbose : bool (default True)
        Whether or not to report the status of the processing.

    Returns
    -------
    coeffs : ~numpy.ndarray, shape of [epochs, channels, frequencies]
        Fourier coefficients of ``data``.

    freqs : ~numpy.ndarray, shape of [frequencies]
        Frequencies (in Hz) in ``coeffs``.
    """
    (n_points, window_func, n_jobs) = _compute_fft_input_checks(
        data, sampling_freq, n_points, window, n_jobs, verbose
    )

    if verbose:
        print("Computing FFT on the data...")

    freqs = (
        sp.fft.rfftfreq(n=n_points, d=1 / sampling_freq)
        .astype(np.float32)  # reduce rounding errors
        .astype(_precision.real)
    )

    window = window_func(data.shape[2])

    loop_kwargs = [
        {"x": sp.signal.detrend(chan_data) * window}
        for chan_data in data.transpose(1, 0, 2)
    ]
    static_kwargs = {"n": n_points}
    coeffs = (
        _compute_in_parallel(
            func=sp.fft.rfft,
            loop_kwargs=loop_kwargs,
            static_kwargs=static_kwargs,
            output=np.zeros(
                (data.shape[1], data.shape[0], freqs.size), dtype=_precision.complex
            ),
            message="Processing channels...",
            n_jobs=n_jobs,
            verbose=verbose,
            prefer="processes",
        )
        .transpose(1, 0, 2)
        .astype(_precision.complex)
    )

    if verbose:
        print("    [FFT computation finished]\n")

    return coeffs, freqs


def _compute_fft_input_checks(
    data: np.ndarray,
    sampling_freq: int | float,
    n_points: int | None,
    window: str,
    n_jobs: int,
    verbose: bool,
) -> tuple[int, Callable, Callable, Callable, int]:
    """Check inputs for computing FFT.

    Returns
    -------
    n_points : int

    window_func : Callable
        Function to use to window the data.

    n_jobs : int
    """
    if not isinstance(data, np.ndarray):
        raise TypeError("`data` must be a NumPy array.")
    if data.ndim != 3:
        raise ValueError("`data` must be a 3D array.")
    if not np.isreal(data).all():
        raise ValueError("`data` must be real-valued.")

    if not isinstance(sampling_freq, (int, float)):
        raise TypeError("`sampling_freq` must be an int or a float.")

    if n_points is None:
        n_points = data.shape[2]
    if not isinstance(n_points, int):
        raise TypeError("`n_points` must be an integer")

    if not isinstance(window, str):
        raise TypeError("`window` must be a str.")
    if window not in ["hanning", "hamming"]:
        raise ValueError("The requested `window` type is not recognised.")
    if window == "hanning":
        window_func = np.hanning
    else:
        window_func = np.hamming

    if not isinstance(n_jobs, int):
        raise TypeError("`n_jobs` must be an integer.")
    if n_jobs < 1 and n_jobs != -1:
        raise ValueError("`n_jobs` must be >= 1 or -1.")
    if n_jobs == -1:
        n_jobs = cpu_count()

    if not isinstance(verbose, bool):
        raise TypeError("`verbose` must be a bool.")

    return n_points, window_func, n_jobs


def compute_tfr(
    data: np.ndarray,
    sampling_freq: int | float,
    freqs: np.ndarray,
    tfr_mode: str = "morlet",
    n_cycles: np.ndarray | int | float = 7.0,
    zero_mean_wavelets: bool | None = None,
    use_fft: bool = True,
    multitaper_time_bandwidth: int | float = 4.0,
    n_jobs: int = 1,
    verbose: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the amplitude time-frequency representation (TFR) of data.

    Parameters
    ----------
    data : ~numpy.ndarray, shape of [epochs, channels, times]
        Real-valued data to compute the amplitude TFR of.

    sampling_freq : int | float
        Sampling frequency (in Hz) of ``data``.

    freqs : ~numpy.ndarray, shape of [frequencies]
        Frequencies (in Hz) to return the TFR for.

    tfr_mode : ``"morlet"`` | ``"multitaper"`` (default ``"morlet"``)
        Mode for computing the TFR. See :func:`mne.time_frequency.tfr_array_morlet` and
        :func:`mne.time_frequency.tfr_array_multitaper`.

    n_cycles : ~numpy.ndarray, shape of [frequencies] | int | float (default ``7.0``)
        Number of cycles in the wavelet when computing the TFR. If an array, the number
        of cycles is given for each frequency, otherwise a fixed value across all
        frequencies is used.

    zero_mean_wavelets : bool | None (default None)
        Whether or not to use wavelets with a mean of 0. If :obj:`None`, the default
        argument of :func:`mne.time_frequency.tfr_array_morlet` and
        :func:`mne.time_frequency.tfr_array_multitaper` is used according to
        ``tfr_mode``.

    use_fft : bool default (True)
        Whether or not to use the fast Fourier transform for convolutions.

    multitaper_time_bandwidth : int | float (default ``4.0``)
        Product between the temporal window length (in seconds) and the frequency
        bandwidth (in Hz). Only used if ``tfr_mode = "multitaper"``. See
        :func:`mne.time_frequency.tfr_array_multitaper` for more information.

    n_jobs : int (default ``1``)
        Number of jobs to run in parallel. If ``-1``, all available CPUs are used.

    verbose : bool (default True)
        Whether or not to report the status of the processing.

    Returns
    -------
    tfr : ~numpy.ndarray, shape of [epochs, channels, frequencies, times]
        Amplitude/power of the TFR of ``data``.

    freqs : ~numpy.ndarray of float, shape of [frequencies]
        Frequencies (in Hz) in ``tfr``.

    Notes
    -----
    This function acts as a wrapper around the MNE TFR computation functions
    :func:`mne.time_frequency.tfr_array_morlet` and
    :func:`mne.time_frequency.tfr_array_multitaper` with ``output = "power"``.
    """
    tfr_func, n_jobs = _compute_tfr_input_checks(
        data,
        sampling_freq,
        freqs,
        tfr_mode,
        n_cycles,
        zero_mean_wavelets,
        use_fft,
        multitaper_time_bandwidth,
        n_jobs,
        verbose,
    )

    tfr_func_kwargs = {
        "data": data,
        "sfreq": sampling_freq,
        "freqs": freqs,
        "n_cycles": n_cycles,
        "use_fft": use_fft,
        "output": "power",
        "n_jobs": n_jobs,
        "verbose": verbose,
    }
    if zero_mean_wavelets is not None:
        tfr_func_kwargs["zero_mean"] = zero_mean_wavelets
    if tfr_mode == "multitaper":
        tfr_func_kwargs["time_bandwidth"] = multitaper_time_bandwidth

    if verbose:
        print("Computing TFR of the data...")

    tfr = np.array(tfr_func(**tfr_func_kwargs), dtype=_precision.real)

    if verbose:
        print("    [TFR computation finished]\n")

    return tfr, freqs.astype(_precision.real)


def _compute_tfr_input_checks(
    data: np.ndarray,
    sampling_freq: int | float,
    freqs: np.ndarray,
    tfr_mode: str,
    n_cycles: np.ndarray | int | float,
    zero_mean_wavelets: bool | None,
    use_fft: bool,
    multitaper_time_bandwidth: int | float,
    n_jobs: int,
    verbose: bool,
) -> tuple[Callable, int]:
    """Check inputs for computing TFR.

    Returns
    -------
    tfr_func
        Function to use to compute TFR.

    n_jobs
    """
    if not isinstance(data, np.ndarray):
        raise TypeError("`data` must be a NumPy array.")
    if data.ndim != 3:
        raise ValueError("`data` must be a 3D array.")

    if not isinstance(sampling_freq, (int, float)):
        raise TypeError("`sampling_freq` must be an int or a float.")

    if not isinstance(freqs, np.ndarray):
        raise TypeError("`freqs` must be a NumPy array.")
    if freqs.ndim != 1:
        raise ValueError("`freqs` must be a 1D array.")
    if freqs.min() < 0 or freqs.max() > sampling_freq * 0.5:
        raise ValueError(
            "Entries of `freqs` must lie in the range [0, Nyquist frequency]."
        )
    if np.all(np.sort(freqs) != freqs):
        raise ValueError("Entries of `freqs` must be in ascending order.")

    tfr_modes = ["morlet", "multitaper"]
    if not isinstance(tfr_mode, str):
        raise TypeError("`tfr_mode` must be a str.")
    if tfr_mode not in tfr_modes:
        raise ValueError(f"`tfr_mode` must be one of {tfr_modes}.")
    if tfr_mode == "morlet":
        tfr_func = time_frequency.tfr_array_morlet
    else:
        tfr_func = time_frequency.tfr_array_multitaper

    if not isinstance(n_cycles, (np.ndarray, int, float)):
        raise TypeError("`n_cycles` must be a NumPy array, an int, or a float.")
    if isinstance(n_cycles, np.ndarray):
        if n_cycles.shape != freqs.shape:
            raise ValueError(
                "If `n_cycles` is an array, it must have the same shape as `freqs`."
            )
        if n_cycles.min() <= 0:
            raise ValueError("Entries of `n_cycles` must be > 0.")
    elif n_cycles <= 0:
        raise ValueError("`n_cycles` must be > 0.")

    if not isinstance(zero_mean_wavelets, bool) and zero_mean_wavelets is not None:
        raise TypeError("`zero_mean_wavelets` must be a bool or None.")

    if not isinstance(use_fft, bool):
        raise TypeError("`use_fft` must be a bool.")

    if tfr_mode == "multitaper":
        if not isinstance(multitaper_time_bandwidth, (int, float)):
            raise TypeError("`multitaper_time_bandwidth` must be an int or a float.")

    if not isinstance(n_jobs, int):
        raise TypeError("`n_jobs` must be an integer.")
    if n_jobs < 1 and n_jobs != -1:
        raise ValueError("`n_jobs` must be >= 1 or -1.")
    if n_jobs == -1:
        n_jobs = cpu_count()

    if not isinstance(verbose, bool):
        raise TypeError("`verbose` must be a bool.")
    if verbose and not np.isreal(data).all():
        warn("`data` is expected to be real-valued.", UserWarning)

    return tfr_func, n_jobs


def compute_rank(data: np.ndarray, sv_tol: int | float = 1e-5) -> int:
    """Compute the minimum rank of data from non-zero singular values.

    Parameters
    ----------
    data : ~numpy.ndarray, shape of [epochs, channels, times]
        Data to find the rank of.

    sv_tol : int | float (default ``1e-5``)
        Tolerance to use to define non-zero singular values, based on the largest
        singular value. Singular values greater than the largest singular value
        multiplied by the tolerance are considered to be non-zero.

    Returns
    -------
    rank : int
        Minimum rank of ``data`` over epochs.
    """
    if not isinstance(data, np.ndarray):
        raise TypeError("`data` must be a NumPy array.")
    if data.ndim != 3:
        raise ValueError("`data` must be a 3D array.")

    if not isinstance(sv_tol, (int, float)):
        raise TypeError("`sv_tol` must be a float or an int.")

    singular_vals = np.linalg.svd(data, compute_uv=False).min(axis=0)

    return np.count_nonzero(singular_vals > singular_vals[0] * sv_tol)


def set_precision(precision: str) -> None:
    """Set the precision of the outputs of PyBispectra classes and functions.

    Attributes
    ----------
    precision : ``"single"`` | ``"double"``
        Precision to use. Accepts ``"single"`` (real values are :obj:`numpy.float32` and
        complex values are :obj:`numpy.complex64`) and ``"double"`` (real values are
        :obj:`numpy.float64` and complex values are :obj:`numpy.complex128`).

    Notes
    -----
    By default, PyBispectra uses double precision. Single precision may be desired to
    increase computational speed and reduce the likelihood of memory errors when
    handling large datasets.
    """
    _precision.set_precision(precision)


# Example/test data ####################################################################

_DATA_ALIAS_FILE_HASH = [
    # PAC
    (
        "sim_data_pac_univariate",
        "sim_data_pac_univariate.npy",
        "sha256:BB0EC7AC8B8578C2BA785D5298DD6F92C554104D441F6DB1E5C2971243FFB4BF",
    ),
    (
        "sim_data_pac_bivariate",
        "sim_data_pac_bivariate.npy",
        "sha256:8A0EAB4F4AFC676FE6DEDEA6410414376257A18703CCE0DA9C0A5848295BD5A6",
    ),
    # TDE
    (
        "sim_data_tde_independent_noise",
        "sim_data_tde_independent_noise.npy",
        "sha256:11C22DC5B5F23F3FA12359A2DA164F3996310EB7E6B43116BAF1935EB38EDF35",
    ),
    (
        "sim_data_tde_correlated_noise",
        "sim_data_tde_correlated_noise.npy",
        "sha256:136B30BA687C6894ABBF4398FA8F22951F5540324209A18668B3B659649CD75B",
    ),
    (
        "sim_data_tde_fbands",
        "sim_data_tde_fbands.npy",
        "sha256:7AF83DED751DCFD23C263E847D92BDE5BDE7AE3250F26CA97A7CD3106D5EA8B8",
    ),
    # Waveshape
    (
        "sim_data_waveshape_peaks_troughs",
        "sim_data_waveshape_peaks_troughs.npy",
        "sha256:5D228E3547EB035F80307FE53D33A852497A3F2256CD87CEC7C926DD5E5A16B8",
    ),
    (
        "sim_data_waveshape_sawtooths",
        "sim_data_waveshape_sawtooths.npy",
        "sha256:7EBD3CF248918E299BEF37D06806A108207F0AE302C6DE57692648BE04DF9743",
    ),
    (
        "sim_data_waveshape_noisy",
        "sim_data_waveshape_noisy.npy",
        "sha256:13D9668FD8574D0ACC972B46F7FDDAE08817FBF8BAD0CA57F4BC4E9703EAFF54",
    ),
]

DATASETS = {alias: filename for alias, filename, _ in _DATA_ALIAS_FILE_HASH}

_pooch = pooch.create(
    path=pooch.os_cache("PyBispectra"),
    base_url="https://github.com/braindatalab/PyBispectra/raw/{__version__}/data/",
    version=__version__,
    version_dev="main",  # if a development version, use the "main" branch
    registry={filename: filehash for _, filename, filehash in _DATA_ALIAS_FILE_HASH},
    retry_if_failed=3,  # try to avoid CI failures
)


def get_example_data_paths(name: str, verbose: bool = True) -> str:
    """Return the path to the requested example data.

    Parameters
    ----------
    name : str
        Name of the example data.

    verbose : bool (default True)
        Whether or not to report the download progress (if the file is not already
        cached).

    Returns
    -------
    path : str
        Path to the example data.

    Notes
    -----
    If the file is not found in the local cache (see :func:`pooch.os_cache` for the
    location), it will be downloaded automatically.
    """
    if name not in DATASETS.keys():
        raise ValueError(f"`name` must be one of: {list(DATASETS.keys())}")

    return _pooch.fetch(fname=DATASETS[name], progressbar=verbose)
