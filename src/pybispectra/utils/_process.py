"""Tools for processing and handling CFC and TDE results."""

from abc import ABC, abstractmethod
from copy import deepcopy
from multiprocessing import cpu_count
from warnings import warn

import numpy as np
from numba import njit

from pybispectra.utils._defaults import _precision
from pybispectra.utils._utils import _fast_find_first, _int_like, _number_like
from pybispectra.utils.results import _ResultsBase


class _ProcessFreqBase(ABC):
    """Base class for processing frequency-domain results."""

    _data_precision: type = _precision.complex

    _data_ndims: tuple = (3, 4)  # [epochs, channels, frequencies (, times)]
    _has_time_dim_placeholder: bool = False

    _indices: tuple = None
    _seeds: tuple = None
    _targets: tuple = None
    _n_cons: int = None

    sampling_freq: float = None
    _f1s: np.ndarray = None
    _f2s: np.ndarray = None
    _times: np.ndarray = None
    _time_idcs: np.ndarray = None

    _n_jobs: int = None

    _results: _ResultsBase = None

    def __init__(
        self,
        data: np.ndarray,
        freqs: np.ndarray,
        sampling_freq: int | float,
        times: np.ndarray | None = None,
        verbose: bool = True,
    ) -> None:
        self._sort_init_inputs(data, freqs, sampling_freq, times, verbose)

    def _sort_init_inputs(
        self,
        data: np.ndarray,
        freqs: np.ndarray,
        sampling_freq: int | float,
        times: np.ndarray | None,
        verbose: bool,
    ) -> None:
        """Check init. inputs are appropriate."""
        if not isinstance(data, np.ndarray):
            raise TypeError("`data` must be a NumPy array.")
        if data.ndim not in self._data_ndims:
            raise ValueError(
                "`data` must be a "
                f"{' or '.join([(str(dim) + 'D') for dim in self._data_ndims])} array."
            )
        assert np.min(self._data_ndims) >= 3 and np.max(self._data_ndims) <= 4, (
            "PyBispectra Internal Error: data to process must be 3D or 4D. Please "
            "contact the PyBispectra developers."
        )

        if not isinstance(freqs, np.ndarray):
            raise TypeError("`freqs` must be a NumPy array.")
        if freqs.ndim != 1:
            raise ValueError("`freqs` must be a 1D array.")
        freqs_diff = np.diff(freqs)
        if not np.allclose(freqs_diff, freqs_diff[0], rtol=1e-3):
            raise ValueError("Entries of `freqs` must be evenly spaced.")

        self._n_epochs, self._n_chans, self._n_freqs = data.shape[:3]

        if self._n_freqs != len(freqs):
            raise ValueError(
                "`data` and `freqs` must contain the same number of frequencies."
            )

        if not isinstance(sampling_freq, _number_like):
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

        if data.ndim == 4:  # Times dimension present
            if times is None:
                times = np.arange(data.shape[3]) / sampling_freq
            else:
                if not isinstance(times, np.ndarray):
                    raise TypeError("`times` must be a NumPy array.")
                if times.ndim != 1:
                    raise ValueError("`times` must be a 1D array.")
                if data.shape[3] != len(times):
                    raise ValueError(
                        "`data` and `times` must contain the same number of timepoints."
                    )
        else:  # Discard times info
            times = None

        if data.ndim == 3 and np.max(self._data_ndims) == 4:
            data = data[..., np.newaxis]  # Add placeholder time dimension
            self._has_time_dim_placeholder = True

        if not isinstance(verbose, bool):
            raise TypeError("`verbose` must be a bool.")

        self._data = np.asarray(data, dtype=self._data_precision)
        self.freqs = np.asarray(freqs, dtype=_precision.real)
        self.times = (
            np.asarray(times, dtype=_precision.real) if times is not None else None
        )
        self.sampling_freq = sampling_freq
        self.verbose = verbose

    def _sort_indices(self, indices: tuple[tuple[int]] | None) -> None:
        """Sort seed-target indices inputs."""
        if indices is None:
            indices = tuple(
                [
                    tuple(np.tile(range(self._n_chans), self._n_chans).tolist()),
                    tuple(np.repeat(range(self._n_chans), self._n_chans).tolist()),
                ]
            )
        if not isinstance(indices, tuple):
            raise TypeError("`indices` must be a tuple.")
        if len(indices) != 2:
            raise ValueError("`indices` must have length of 2.")
        self._indices = indices

        seeds = indices[0]
        targets = indices[1]
        for group_idcs in (seeds, targets):
            if not isinstance(group_idcs, tuple):
                raise TypeError("Entries of `indices` must be tuples.")
            if any(not isinstance(idx, _int_like) for idx in group_idcs):
                raise TypeError(
                    "Entries for seeds and targets in `indices` must be ints."
                )
            if any(idx < 0 or idx >= self._n_chans for idx in group_idcs):
                raise ValueError(
                    "`indices` contains indices for channels not present in the data."
                )
        if len(seeds) != len(targets):
            raise ValueError("Entries of `indices` must have equal length.")
        self._seeds = seeds
        self._targets = targets

        self._n_cons = len(seeds)

    def _sort_freqs(
        self, f1s: tuple[int | float] | None, f2s: tuple[int | float] | None
    ) -> None:
        """Sort frequency inputs."""
        check_f1s = True
        check_f2s = True
        if f1s is None:
            self._f1s = self.freqs
            check_f1s = False
        if f2s is None:
            self._f2s = self.freqs
            check_f2s = False

        for freqs, check_freqs in zip([f1s, f2s], [check_f1s, check_f2s]):
            if check_freqs:
                if not isinstance(freqs, tuple):
                    raise TypeError("`f1s` and `f2s` must be tuples.")
                if len(freqs) != 2:
                    raise ValueError("`f1s` and `f2s` must have lengths of 2.")
                if any(freq < 0 for freq in freqs):
                    raise ValueError("Entries of `f1s` and `f2s` must be >= 0.")
                if any(freq > self.sampling_freq / 2 for freq in freqs):
                    raise ValueError(
                        "Entries of `f1s` and `f2s` must be <= the Nyquist frequency."
                    )

        if check_f1s:
            f1_idcs = np.argwhere((self.freqs >= f1s[0]) & (self.freqs <= f1s[1])).T[0]
            if f1_idcs.size == 0:
                raise ValueError(
                    "No frequencies are present in the data for the range in `f1s`."
                )
            self._f1s = self.freqs[f1_idcs]
        if check_f2s:
            f2_idcs = np.argwhere((self.freqs >= f2s[0]) & (self.freqs <= f2s[1])).T[0]
            if f2_idcs.size == 0:
                raise ValueError(
                    "No frequencies are present in the data for the range in `f2s`."
                )
            self._f2s = self.freqs[f2_idcs]

        if self.verbose:
            if self._f1s.max() >= self._f2s.min():
                warn(
                    "At least one value in `f1s` is >= a value in `f2s`. The "
                    "corresponding result(s) will have a value of NaN.",
                    UserWarning,
                )

    def _sort_tmin_tmax(self, times: tuple[int | float] | None) -> None:
        """Sort time range inputs."""
        if times is None:
            times = (-np.inf, np.inf)

        if not isinstance(times, tuple):
            raise TypeError("`times` must be a tuple or None.")
        if len(times) != 2:
            raise ValueError("`times` must have length of 2.")

        for time in times:
            if not isinstance(time, _number_like):
                raise TypeError("Entries of `times` must be int or float.")

        if self.times is None:
            timepoints = np.array([0])
        else:
            timepoints = self.times

        self._time_idcs = np.argwhere(
            (timepoints >= times[0]) & (timepoints <= times[1])
        ).T[0]
        if self._time_idcs.size == 0:
            raise ValueError(
                "No timepoints are present in the data for the range in `times`."
            )
        self._times = timepoints[self._time_idcs]

    def _sort_parallelisation(self, n_jobs: int) -> None:
        """Sort parallelisation inputs."""
        if not isinstance(n_jobs, _int_like):
            raise TypeError("`n_jobs` must be an integer.")
        if n_jobs < 1 and n_jobs != -1:
            raise ValueError("`n_jobs` must be >= 1 or -1.")
        if n_jobs == -1:
            n_jobs = cpu_count()

        self._n_jobs = n_jobs

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
        self._times = None
        self._time_idcs = None

        self._n_jobs = None

        self._results = None

    @abstractmethod
    def _store_results(self) -> None:
        """Store computed results in an object."""

    @property
    @abstractmethod
    def results(self) -> None:  # pragma: no cover
        pass

    @property
    def data(self) -> np.ndarray:
        if self._has_time_dim_placeholder:
            return self._data[..., 0]
        return self._data

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
                any(seed == target for seed, target in zip(self._seeds, self._targets))
            ):
                warn(
                    "The seed and target for at least one connection is the same "
                    "channel. The corresponding antisymmetrised result(s) will be "
                    "NaN-valued.",
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
                    "At least one value of `f2s` + `f1s` is not present in the "
                    "frequencies. The corresponding result(s) will be NaN-valued.",
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
    data : numpy.ndarray of float, shape of [epochs, 2, frequencies, times]
        FFT coefficients, where the second dimension contains the data for the seed and
        target channel of a single connection, respectively.

    freqs : numpy.ndarray of float, shape of [frequencies]
        Frequencies in ``data``.

    f1s : numpy.ndarray of float, shape of [low frequencies]
        Low frequencies to compute the bispectrum for.

    f2s : numpy.ndarray of float, shape of [high frequencies]
        High frequencies to compute the bispectrum for.

    kmn : numpy.ndarray of int, shape of [x, 3]
        Array of variable length (x) of arrays, where each sub-array contains the k, m,
        and n channel indices in ``data``, respectively, to compute the bispectrum for.

    precision : type
        Precision to use for the computation. Either ``numpy.complex64`` (single) or
        ``numpy.complex128`` (double).

    Returns
    -------
    results : numpy.ndarray of complex float, shape of [x, f1s, f2s, times]
        Complex-valued array containing the bispectrum of a single connection, where the
        first dimension corresponds to the different channel indices given in ``kmn``.

    Notes
    -----
    No checks on the input data are performed for speed.
    """
    results = np.full(
        (kmn.shape[0], f1s.shape[0], f2s.shape[0], data.shape[3]),
        fill_value=np.nan + np.nan * 1j,
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
            fdiff_fi = f1_fi + f2_fi
            if f1 <= f2 and fdiff_fi < freqs.size:
                for kmn_i, (k, m, n) in enumerate(kmn):
                    if np.isnan(results[kmn_i, f1_ri, f2_ri]).all():
                        results[kmn_i, f1_ri, f2_ri] = 0 + 0j
                    for epoch_data in data:
                        results[kmn_i, f1_ri, f2_ri] += (
                            epoch_data[k, f1_fi]
                            * epoch_data[m, f2_fi]
                            * np.conjugate(epoch_data[n, fdiff_fi])
                        )

    return np.divide(results, data.shape[0]).astype(precision)


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

    Parameters
    ----------
    data : numpy.ndarray of float, shape of [epochs, 2, frequencies, times]
        FFT coefficients, where the second dimension contains the data for the seed and
        target channel of a single connection, respectively.

    freqs : numpy.ndarray of float, shape of [frequencies]
        Frequencies in ``data``.

    f1s : numpy.ndarray of float, shape of [low frequencies]
        Low frequencies to compute the threenorm for.

    f2s : numpy.ndarray of float, shape of [high frequencies]
        High frequencies to compute the threenorm for.

    kmn : numpy.ndarray of int, shape of [x, 3]
        Array of variable length (x) of arrays, where each sub-array contains the k, m,
        and n channel indices in ``data``, respectively, to compute the threenorm for.

    precision : type
        Precision to use for the computation. Either ``numpy.complex64`` (single) or
        ``numpy.complex128`` (double).

    Returns
    -------
    results : numpy.ndarray of float, shape of [x, f1s, f2s, times]
        Threenorm of a single connection, where the first dimension corresponds to the
        different channel indices given in ``kmn``.

    Notes
    -----
    No checks on the input data are performed for speed.
    """
    results = np.full(
        (kmn.shape[0], f1s.shape[0], f2s.shape[0], data.shape[3]),
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
            fdiff_fi = f1_fi + f2_fi
            if f1 <= f2 and fdiff_fi < freqs.size:
                for kmn_i, (k, m, n) in enumerate(kmn):
                    results[kmn_i, f1_ri, f2_ri] = (
                        (np.abs(data[:, k, f1_fi]) ** 3).mean()
                        * (np.abs(data[:, m, f2_fi]) ** 3).mean()
                        * (np.abs(data[:, n, fdiff_fi]) ** 3).mean()
                    ) ** (1 / 3)

    return results
