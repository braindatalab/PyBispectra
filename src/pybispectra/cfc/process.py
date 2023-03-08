"""Tools for processing and handling cross-freq. coupling results."""

import copy
from abc import ABC, abstractmethod
from warnings import warn

import numpy as np

from pybispectra.utils import Results


class _Process(ABC):
    """Base class for processing results.

    PARAMETERS
    ----------
    data : NumPy ndarray
    -   FFT coefficients with shape [epochs x channels x frequencies].

    freqs : NumPy ndarray
    -   1D array of the frequencies in `data`.

    verbose : bool; default True
    -   Whether or not to report the progress of the processing.

    METHODS
    -------
    compute
    -   Compute results.

    get_results
    -   Return a copy of the results.

    copy
    -   Return a copy of the object.

    ATTRIBUTES
    ----------
    data : NumPy ndarray
    -   3D array of FFT coefficients with shape [epochs x channels x
        frequencies].

    freqs : NumPy ndarray
    -   1D array of the frequencies in `data`.

    indices : tuple of NumPy ndarray
    -   2 arrays containing the seed and target indices (respectively) most
        recently used with `compute`.

    f1 : NumPy ndarray
    -   1D array of low frequencies most recently used with `compute`.

    f2 : NumPy ndarray
    -   1D array of high frequencies most recently used with `compute`.

    verbose : bool; default True
    -   Whether or not to report the progress of the processing.
    """

    indices = None
    _seeds = None
    _targets = None
    _n_cons = None

    f1 = None
    f2 = None

    _n_jobs = None

    _results = None

    def __init__(
        self, data: np.ndarray, freqs: np.ndarray, verbose: bool = True
    ) -> None:
        self.verbose = copy.copy(verbose)
        self._sort_init_inputs(data, freqs)

    def _sort_init_inputs(self, data: np.ndarray, freqs: np.ndarray) -> None:
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
                "frequencies"
            )

        self.data = data.copy()
        self.freqs = freqs.copy()

    def _sort_indices(self, indices: tuple[np.ndarray] | None) -> None:
        """Sort seed-target indices inputs."""
        indices = copy.copy(indices)
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
        self.indices = copy.deepcopy(indices)

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
        self, f1: np.ndarray | None, f2: np.ndarray | None
    ) -> None:
        """Sort frequency inputs."""
        if f1 is None:
            f1 = self.freqs.copy()[:-1]
            if f2 is None:
                f2 = self.freqs.copy()[1:]
        if f2 is None:
            f2 = f1[1:].copy()

        if not isinstance(f1, np.ndarray) or not isinstance(f2, np.ndarray):
            raise TypeError("`f1` and `f2` must be NumPy ndarrays.")
        if f1.ndim != 1 or f2.ndim != 1:
            raise ValueError("`f1` and `f2` must be 1D arrays.")

        if any(freq not in self.freqs for freq in f1) or any(
            freq not in self.freqs for freq in f2
        ):
            raise ValueError(
                "All frequencies in `f1` and `f2` must be present in the "
                "data."
            )

        if self.verbose:
            if any(lfreq >= hfreq for hfreq in f2 for lfreq in f1):
                warn(
                    "At least one value in `f1` is >= a value in `f2`. The "
                    "corresponding result(s) will have a value of NaN.",
                    UserWarning,
                )

        self.f1 = f1.copy()
        self.f2 = f2.copy()

    def _sort_parallelisation(self, n_jobs: int) -> None:
        """Sort parallelisation inputs."""
        if not isinstance(n_jobs, int):
            raise TypeError("`n_jobs` must be an integer.")

        if n_jobs < 1:
            raise ValueError("`n_jobs` must be >= 1.")

        self._n_jobs = copy.copy(n_jobs)

    @abstractmethod
    def compute(self):
        """Compute results."""

    def _reset_attrs(self) -> None:
        """Reset attrs. of the object to prevent interference."""
        self.indices = None
        self._seeds = None
        self._targets = None
        self._n_cons = None

        self.f1 = None
        self.f2 = None

        self._n_jobs = None

        self._results = None

    @abstractmethod
    def _store_results(self) -> None:
        """Store computed results in an object."""

    @property
    def results(self) -> Results | tuple[Results]:
        """Return a copy of the results."""
        return self._results

    def copy(self):
        """Return a copy of the object."""
        return copy.deepcopy(self)
