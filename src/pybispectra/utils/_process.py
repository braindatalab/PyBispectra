"""Tools for processing and handling CFC and TDE results."""

from abc import ABC, abstractmethod, abstractproperty
from copy import deepcopy
from multiprocessing import cpu_count
from warnings import warn

from numba import njit
import numpy as np

from pybispectra.utils import fast_find_first


class _ProcessFreqBase(ABC):
    """Base class for processing frequency-domain results."""

    indices = None
    _seeds = None
    _targets = None
    _n_cons = None

    sampling_freq = None
    f1s = None
    f2s = None

    _n_jobs = None

    _results = None

    def __init__(
        self,
        data: np.ndarray,
        freqs: np.ndarray,
        sampling_freq: int | float,
        verbose: bool = True,
    ) -> None:
        self.verbose = deepcopy(verbose)
        self._sort_init_inputs(data, freqs, sampling_freq)

    def _sort_init_inputs(
        self, data: np.ndarray, freqs: np.ndarray, sampling_freq: int | float
    ) -> None:
        """Check init. inputs are appropriate."""
        if not isinstance(data, np.ndarray):
            raise TypeError("`data` must be a NumPy array.")
        if data.ndim != 3:
            raise ValueError("`data` must be a 3D array.")

        if not isinstance(freqs, np.ndarray):
            raise TypeError("`freqs` must be a NumPy array.")
        if freqs.ndim != 1:
            raise ValueError("`freqs` must be a 1D array.")

        self._n_epochs, self._n_chans, self._n_freqs = data.shape

        if self._n_freqs != len(freqs):
            raise ValueError(
                "`data` and `freqs` should contain the same number of "
                "frequencies."
            )

        if not isinstance(sampling_freq, (int, float)):
            raise TypeError("`sampling_freq` must be an int or a float.")
        if np.abs(freqs).max() * 2 > sampling_freq:
            raise ValueError(
                "At least one entry of `freqs` is > the Nyquist frequency."
            )

        self.data = data.copy()
        self.freqs = freqs.copy()
        self.sampling_freq = deepcopy(sampling_freq)

    def _sort_indices(self, indices: tuple[np.ndarray] | None) -> None:
        """Sort seed-target indices inputs."""
        indices = deepcopy(indices)
        if indices is None:
            indices = tuple(
                [
                    np.tile(range(self._n_chans), self._n_chans),
                    np.repeat(range(self._n_chans), self._n_chans),
                ]
            )
        if not isinstance(indices, tuple):
            raise TypeError("`indices` should be a tuple.")
        if len(indices) != 2:
            raise ValueError("`indices` should have a length of 2.")
        self.indices = deepcopy(indices)

        seeds = indices[0]
        targets = indices[1]
        for group_idcs in (seeds, targets):
            if not isinstance(group_idcs, np.ndarray):
                raise TypeError("Entries of `indices` should be NumPy arrays.")
            if any(idx < 0 or idx >= self._n_chans for idx in group_idcs):
                raise ValueError(
                    "`indices` contains indices for channels not present in "
                    "the data."
                )
        if len(seeds) != len(targets):
            raise ValueError("Entires of `indices` must have equal length.")
        self._seeds = seeds
        self._targets = targets

        self._n_cons = len(seeds)

    def _sort_freqs(
        self, f1s: np.ndarray | None, f2s: np.ndarray | None
    ) -> None:
        """Sort frequency inputs."""
        if f1s is None:
            f1s = self.freqs.copy()
        if f2s is None:
            f2s = f1s.copy()

        if not isinstance(f1s, np.ndarray) or not isinstance(f2s, np.ndarray):
            raise TypeError("`f1s` and `f2s` must be NumPy arrays.")
        if f1s.ndim != 1 or f2s.ndim != 1:
            raise ValueError("`f1s` and `f2s` must be 1D arrays.")

        if any(freq not in self.freqs for freq in f1s) or any(
            freq not in self.freqs for freq in f2s
        ):
            raise ValueError(
                "All frequencies in `f1s` and `f2s` must be present in the "
                "data."
            )

        if self.sampling_freq is not None:
            if self.sampling_freq < f2s[-1] * 2:
                raise ValueError("`sampling_freq` must be >= all entries of f2s * 2.")

        if self.verbose:
            if any(lfreq >= hfreq for hfreq in f2s for lfreq in f1s):
                warn(
                    "At least one value in `f1s` is >= a value in `f2s`. The "
                    "corresponding result(s) will have a value of NaN.",
                    UserWarning,
                )

        self.f1s = f1s.copy()
        self.f2s = f2s.copy()

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
        self.indices = None
        self._seeds = None
        self._targets = None
        self._n_cons = None

        self.f1s = None
        self.f2s = None

        self._n_jobs = None

        self._results = None

    @abstractmethod
    def _store_results(self) -> None:
        """Store computed results in an object."""

    @abstractproperty
    def results(self) -> None:
        """Return a copy of the results."""

    def copy(self):
        """Return a copy of the object."""
        return deepcopy(self)


class _ProcessBispectrum(_ProcessFreqBase):
    """Base class for processing bispectrum-based results."""

    def _sort_indices(self, indices: np.ndarray) -> None:
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

    def _sort_freqs(self, f1s: np.ndarray, f2s: np.ndarray) -> None:
        """Sort frequency inputs."""
        super()._sort_freqs(f1s, f2s)

        if self.verbose:
            if any(
                hfreq + lfreq not in self.freqs
                for hfreq in self.f2s
                for lfreq in self.f1s
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
) -> np.ndarray:
    """Compute the bispectrum for a single connection.

    Parameters
    ----------
    data : np.ndarray of float, shape of [epochs, 2, frequencies]
        FFT coefficients, where the second dimension contains the data for the
        seed and target channel of a single connection, respectively.

    freqs : np.ndarray of float, shape of [frequencies]
        Frequencies in ``data``.

    f1s : np.ndarray of float, shape of [frequencies]
        Low frequencies to compute the bispectrum for.

    f2s : np.ndarray of float, shape of [frequencies]
        High frequencies to compute the bispectrum for.

    kmn : np.ndarray of int, shape of [x, 3]
        Array of variable length (x) of arrays, where each sub-array contains
        the k, m, and n channel indices in ``data``, respectively, to compute
        the bispectrum for.

    Returns
    -------
    results : np.ndarray of complex float, shape of [x, epochs, f1s, f2s]
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
        dtype=np.complex128,
    )
    for f1_i, f1 in enumerate(f1s):
        for f2_i, f2 in enumerate(f2s):
            if f1 < f2 and (f2 + f1) in freqs:
                f1_loc = fast_find_first(freqs, f1)
                f2_loc = fast_find_first(freqs, f2)
                fdiff_loc = fast_find_first(freqs, f2 + f1)
                for kmn_i, (k, m, n) in enumerate(kmn):
                    for epoch_i, epoch_data in enumerate(data):
                        results[kmn_i, epoch_i, f1_i, f2_i] = (
                            epoch_data[k, f1_loc]
                            * epoch_data[m, f2_loc]
                            * np.conjugate(epoch_data[n, fdiff_loc])
                        )

    return results


@njit
def _compute_threenorm(
    data: np.ndarray,
    freqs: np.ndarray,
    f1s: np.ndarray,
    f2s: np.ndarray,
) -> np.ndarray:
    """Compute threenorm for a single connection across epochs.

    PARAMETERS
    ----------
    data : numpy.ndarray of float, shape of [epochs, 2, frequencies]
        FFT coefficients, where the second dimension contains the data for the
        seed and target channel of a single connection, respectively.

    freqs : numpy.ndarray of float, shape of [frequencies]
        Frequencies in ``data``.

    f1s : numpy.ndarray of float, shape of [frequencies]
        Low frequencies to compute the threenorm for.

    f2s : numpy.ndarray of float, shape of [frequencies]
        High frequencies to compute the threenorm for.

    RETURNS
    -------
    results : numpy.ndarray of float, shape of [f1s, f2s]
        Threenorm of a single connection averaged across epochs.
    """
    results = np.full(
        (f1s.shape[0], f2s.shape[0]), fill_value=np.nan, dtype=np.float64
    )
    for f1_i, f1 in enumerate(f1s):
        for f2_i, f2 in enumerate(f2s):
            if f1 < f2 and (f2 + f1) in freqs:
                fft_f1 = data[:, 0, fast_find_first(freqs, f1)]
                fft_f2 = data[:, 1, fast_find_first(freqs, f2)]
                fft_fdiff = data[:, 1, fast_find_first(freqs, f2 + f1)]
                results[f1_i, f2_i] = (
                    (np.abs(fft_f1) ** 3).mean()
                    * (np.abs(fft_f2) ** 3).mean()
                    * (np.abs(fft_fdiff) ** 3).mean()
                ) ** 1 / 3

    return results
