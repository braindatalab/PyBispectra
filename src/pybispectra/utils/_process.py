"""Tools for processing and handling CFC and TDE results."""

from abc import ABC, abstractmethod
from copy import deepcopy
from multiprocessing import cpu_count
from warnings import warn

from numba import njit
import numpy as np

from pybispectra.utils.results import _ResultsBase
from pybispectra.utils._defaults import _precision
from pybispectra.utils._utils import _fast_find_first


class _ProcessFreqBase(ABC):
    """Base class for processing frequency-domain results."""

    _data_precision: type = _precision.complex

    _data_ndims: int = 3  # usu. [epochs, channels, frequencies, (times)]

    _indices: tuple = None
    _seeds: tuple = None
    _targets: tuple = None
    _n_cons: int = None

    sampling_freq: float = None
    _f1s: np.ndarray = None
    _f2s: np.ndarray = None

    _n_jobs: int = None

    _results: _ResultsBase = None

    def __init__(
        self,
        data: np.ndarray,
        freqs: np.ndarray,
        sampling_freq: int | float,
        verbose: bool = True,
    ) -> None:
        self._sort_init_inputs(data, freqs, sampling_freq, verbose)

    def _sort_init_inputs(
        self,
        data: np.ndarray,
        freqs: np.ndarray,
        sampling_freq: int | float,
        verbose: bool,
    ) -> None:
        """Check init. inputs are appropriate."""
        if not isinstance(data, np.ndarray):
            raise TypeError("`data` must be a NumPy array.")
        if data.ndim != self._data_ndims:
            raise ValueError(f"`data` must be a {self._data_ndims}D array.")

        if not isinstance(freqs, np.ndarray):
            raise TypeError("`freqs` must be a NumPy array.")
        if freqs.ndim != 1:
            raise ValueError("`freqs` must be a 1D array.")

        self._n_epochs, self._n_chans, self._n_freqs = data.shape[:3]

        if self._n_freqs != len(freqs):
            raise ValueError(
                "`data` and `freqs` must contain the same number of "
                "frequencies."
            )

        if not isinstance(sampling_freq, (int, float)):
            raise TypeError("`sampling_freq` must be an int or a float.")
        if np.abs(freqs).max() > sampling_freq / 2:
            raise ValueError(
                "At least one entry of `freqs` is > the Nyquist frequency."
            )

        if np.any(freqs < 0):
            raise ValueError("Entries of `freqs` must be >= 0.")

        max_freq_idx = np.where(freqs == np.abs(freqs).max())[0][0]
        if max_freq_idx == 0 or np.any(
            freqs[:max_freq_idx] != np.sort(freqs[:max_freq_idx])
        ):
            raise ValueError("Entries of `freqs` must be in ascending order.")

        if not isinstance(verbose, bool):
            raise TypeError("`verbose` must be a bool.")

        self.data = data.copy().astype(self._data_precision)
        self.freqs = freqs.copy().astype(_precision.real)
        self.sampling_freq = deepcopy(sampling_freq)
        self.verbose = deepcopy(verbose)

    def _sort_indices(self, indices: tuple[tuple[int]] | None) -> None:
        """Sort seed-target indices inputs."""
        indices = deepcopy(indices)
        if indices is None:
            indices = tuple(
                [
                    tuple(
                        np.tile(range(self._n_chans), self._n_chans).tolist()
                    ),
                    tuple(
                        np.repeat(range(self._n_chans), self._n_chans).tolist()
                    ),
                ]
            )
        if not isinstance(indices, tuple):
            raise TypeError("`indices` must be a tuple.")
        if len(indices) != 2:
            raise ValueError("`indices` must have length of 2.")
        self._indices = deepcopy(indices)

        seeds = indices[0]
        targets = indices[1]
        for group_idcs in (seeds, targets):
            if not isinstance(group_idcs, tuple):
                raise TypeError("Entries of `indices` must be tuples.")
            if any(not isinstance(idx, int) for idx in group_idcs):
                raise TypeError(
                    "Entries for seeds and targets in `indices` must be ints."
                )
            if any(idx < 0 or idx >= self._n_chans for idx in group_idcs):
                raise ValueError(
                    "`indices` contains indices for channels not present in "
                    "the data."
                )
        if len(seeds) != len(targets):
            raise ValueError("Entries of `indices` must have equal length.")
        self._seeds = deepcopy(seeds)
        self._targets = deepcopy(targets)

        self._n_cons = len(seeds)

    def _sort_freqs(
        self, f1s: tuple[int | float] | None, f2s: tuple[int | float] | None
    ) -> None:
        """Sort frequency inputs."""
        check_f1s = True
        check_f2s = True
        if f1s is None:
            self._f1s = self.freqs.copy()
            check_f1s = False
        if f2s is None:
            self._f2s = self.freqs.copy()
            check_f2s = False

        for freqs, check_freqs in zip([f1s, f2s], [check_f1s, check_f2s]):
            if check_freqs:
                if not isinstance(freqs, tuple):
                    raise TypeError("`f1s` and `f2s` must be tuples.")
                if len(freqs) != 2:
                    raise ValueError("`f1s` and `f2s` must have lengths of 2.")
                if any(freq < 0 for freq in freqs):
                    raise ValueError(
                        "Entries of `f1s` and `f2s` must be >= 0."
                    )
                if any(freq > self.sampling_freq / 2 for freq in freqs):
                    raise ValueError(
                        "Entries of `f1s` and `f2s` must be <= the Nyquist "
                        "frequency."
                    )

        if check_f1s:
            f1_idcs = np.argwhere(
                (self.freqs >= f1s[0]) & (self.freqs <= f1s[1])
            ).T[0]
            if f1_idcs.size == 0:
                raise ValueError(
                    "No frequencies are present in the data for the range in "
                    "`f1s`."
                )
            self._f1s = self.freqs[f1_idcs].copy()
        if check_f2s:
            f2_idcs = np.argwhere(
                (self.freqs >= f2s[0]) & (self.freqs <= f2s[1])
            ).T[0]
            if f2_idcs.size == 0:
                raise ValueError(
                    "No frequencies are present in the data for the range in "
                    "`f2s`."
                )
            self._f2s = self.freqs[f2_idcs].copy()

        if self.verbose:
            if self._f1s.max() >= self._f2s.min():
                warn(
                    "At least one value in `f1s` is >= a value in `f2s`. The "
                    "corresponding result(s) will have a value of NaN.",
                    UserWarning,
                )

    def _sort_parallelisation(self, n_jobs: int) -> None:
        """Sort parallelisation inputs."""
        if not isinstance(n_jobs, int):
            raise TypeError("`n_jobs` must be an integer.")
        if n_jobs < 1 and n_jobs != -1:
            raise ValueError("`n_jobs` must be >= 1 or -1.")
        if n_jobs == -1:
            n_jobs = cpu_count()

        self._n_jobs = deepcopy(n_jobs)

    @abstractmethod
    def compute(self):
        """Compute results."""

    def _reset_attrs(self) -> None:
        """Reset attrs. of the object to prevent interference."""
        self._indices = None
        self._seeds = None
        self._targets = None
        self._n_cons = None

        self._f1s = None
        self._f2s = None

        self._n_jobs = None

        self._results = None

    @abstractmethod
    def _store_results(self) -> None:
        """Store computed results in an object."""

    @property
    @abstractmethod
    def results(self) -> None:
        """Return a copy of the results."""

    def copy(self):
        """Return a copy of the object."""
        return deepcopy(self)


class _ProcessBispectrum(_ProcessFreqBase):
    """Base class for processing bispectrum-based results."""

    _bispectrum: np.ndarray = None
    _threenorm: np.ndarray = None
    _bicoherence: np.ndarray = None

    _return_antisym: bool = None

    def _sort_indices(self, indices: tuple[tuple[int]]) -> None:
        """Sort seed-target indices inputs."""
        super()._sort_indices(indices)

        if self.verbose:
            if self._return_antisym and (
                any(
                    seed == target
                    for seed, target in zip(self._seeds, self._targets)
                )
            ):
                warn(
                    "The seed and target for at least one connection is the "
                    "same channel. The corresponding antisymmetrised "
                    "result(s) will be NaN-valued.",
                    UserWarning,
                )

    def _sort_freqs(
        self, f1s: tuple[int | float] | None, f2s: tuple[int | float] | None
    ) -> None:
        """Sort frequency inputs."""
        super()._sort_freqs(f1s, f2s)

        if self.verbose:
            if any(
                hfreq + lfreq not in self.freqs
                for hfreq in self._f2s
                for lfreq in self._f1s
            ):
                warn(
                    "At least one value of `f2s` + `f1s` is not present in "
                    "the frequencies. The corresponding result(s) will be "
                    "NaN-valued.",
                    UserWarning,
                )


@njit
def _compute_bispectrum(
    data: np.ndarray,
    freqs: np.ndarray,
    f1s: np.ndarray,
    f2s: np.ndarray,
    kmn: np.ndarray,
    precision: type,
) -> np.ndarray:  # pragma: no cover
    """Compute the bispectrum for a single connection.

    Parameters
    ----------
    data : numpy.ndarray of float, shape of [epochs, 2, frequencies]
        FFT coefficients, where the second dimension contains the data for the
        seed and target channel of a single connection, respectively.

    freqs : numpy.ndarray of float, shape of [frequencies]
        Frequencies in ``data``.

    f1s : numpy.ndarray of float, shape of [low frequencies]
        Low frequencies to compute the bispectrum for.

    f2s : numpy.ndarray of float, shape of [high frequencies]
        High frequencies to compute the bispectrum for.

    kmn : numpy.ndarray of int, shape of [x, 3]
        Array of variable length (x) of arrays, where each sub-array contains
        the k, m, and n channel indices in ``data``, respectively, to compute
        the bispectrum for.

    precision : type
        Precision to use for the computation. Either ``numpy.complex64``
        (single) or ``numpy.complex128`` (double).

    Returns
    -------
    results : numpy.ndarray of complex float, shape of [x, epochs, f1s, f2s]
        Complex-valued array containing the bispectrum of a single connection,
        where the first dimension corresponds to the different channel indices
        given in ``kmn``.

    Notes
    -----
    Averaging across epochs is not performed here as ``numpy.mean`` of complex
    numbers is not supported when compiling using Numba.

    No checks on the input data are performed for speed.
    """
    results = np.full(
        (len(kmn), data.shape[0], f1s.shape[0], f2s.shape[0]),
        fill_value=np.nan,
        dtype=precision,
    )
    f1_start = _fast_find_first(freqs, f1s[0], 0)
    f1_end = _fast_find_first(freqs, f1s[-1], f1_start)
    f2_start = _fast_find_first(freqs, f2s[0], 0)
    f2_end = _fast_find_first(freqs, f2s[-1], f2_start)
    for f1_ri, f1_fi in enumerate(range(f1_start, f1_end + 1)):
        f1 = freqs[f1_fi]
        for f2_ri, f2_fi in enumerate(range(f2_start, f2_end + 1)):
            f2 = freqs[f2_fi]
            if f1 <= f2 and f2 + f1 in freqs:
                fdiff_fi = _fast_find_first(freqs, f2 + f1, f2_fi + f1_fi)
                for kmn_i, (k, m, n) in enumerate(kmn):
                    for epoch_i in range(data.shape[0]):
                        results[kmn_i, epoch_i, f1_ri, f2_ri] = (
                            data[epoch_i, k, f1_fi]
                            * data[epoch_i, m, f2_fi]
                            * np.conjugate(data[epoch_i, n, fdiff_fi])
                        )

    return results


@njit
def _compute_threenorm(
    data: np.ndarray,
    freqs: np.ndarray,
    f1s: np.ndarray,
    f2s: np.ndarray,
    kmn: np.ndarray,
    precision: type,
) -> np.ndarray:  # pragma: no cover
    """Compute threenorm for a single connection across epochs.

    PARAMETERS
    ----------
    data : numpy.ndarray of float, shape of [epochs, 2, frequencies]
        FFT coefficients, where the second dimension contains the data for the
        seed and target channel of a single connection, respectively.

    freqs : numpy.ndarray of float, shape of [frequencies]
        Frequencies in ``data``.

    f1s : numpy.ndarray of float, shape of [low frequencies]
        Low frequencies to compute the threenorm for.

    f2s : numpy.ndarray of float, shape of [high frequencies]
        High frequencies to compute the threenorm for.

    kmn : numpy.ndarray of int, shape of [x, 3]
        Array of variable length (x) of arrays, where each sub-array contains
        the k, m, and n channel indices in ``data``, respectively, to compute
        the threenorm for.

    precision : type
        Precision to use for the computation. Either ``numpy.complex64``
        (single) or ``numpy.complex128`` (double).

    RETURNS
    -------
    results : numpy.ndarray of float, shape of [x, f1s, f2s]
        Threenorm of a single connection, where the first dimension corresponds
        to the different channel indices given in ``kmn``.
    """
    results = np.full(
        (len(kmn), f1s.shape[0], f2s.shape[0]),
        fill_value=np.nan,
        dtype=precision,
    )
    f1_start = _fast_find_first(freqs, f1s[0], 0)
    f1_end = _fast_find_first(freqs, f1s[-1], f1_start)
    f2_start = _fast_find_first(freqs, f2s[0], 0)
    f2_end = _fast_find_first(freqs, f2s[-1], f2_start)
    for f1_ri, f1_fi in enumerate(range(f1_start, f1_end + 1)):
        f1 = freqs[f1_fi]
        for f2_ri, f2_fi in enumerate(range(f2_start, f2_end + 1)):
            f2 = freqs[f2_fi]
            if f1 <= f2 and f2 + f1 in freqs:
                fdiff_fi = _fast_find_first(freqs, f2 + f1, f2_fi + f1_fi)
                for kmn_i, (k, m, n) in enumerate(kmn):
                    results[kmn_i, f1_ri, f2_ri] = (
                        (np.abs(data[:, k, f1_fi]) ** 3).mean()
                        * (np.abs(data[:, m, f2_fi]) ** 3).mean()
                        * (np.abs(data[:, n, fdiff_fi]) ** 3).mean()
                    ) ** (1 / 3)

    return results
