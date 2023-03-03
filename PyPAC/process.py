"""Tools for processing and handling cross-freq. coupling results."""

import copy
from abc import ABC, abstractmethod
from warnings import warn

import numpy as np
import scipy as sp


class Process(ABC):
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
    -   FFT coefficients with shape [epochs x channels x frequencies].

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
            if any(idx >= self._n_chans for idx in group_idcs):
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
                "All frequencies in `f1` and `f2` must be present in the data."
            )

        if self.verbose:
            if any(lfreq >= hfreq for hfreq in f2 for lfreq in f1):
                warn(
                    "At least one value in `f1` is >= a value in `f2`. The "
                    "corresponding result(s) will have a value of NaN.\n",
                    UserWarning,
                )

        self.f1 = f1.copy()
        self.f2 = f2.copy()

    @abstractmethod
    def compute(self):
        """Compute results."""

    @abstractmethod
    def get_results(self):
        """Return a copy of the results."""

    def _get_compact_results(
        self, results: np.ndarray
    ) -> tuple[np.ndarray, tuple[np.ndarray]]:
        """Return a compacted form of the results.

        PARAMETERS
        ----------
        results : NumPy ndarray
        -   Results with shape [connections x f2 x f1].

        RETURNS
        -------
        compact_results : NumPy ndarray
        -   Results with shape [seeds x targets x f2 x f1].

        indices : tuple[NumPy ndarray]
        -   Channel indices of `compact_results`, for the seeds and targets,
            respectively.
        """
        fill = np.full((results.shape[1], results.shape[2]), fill_value=np.nan)
        compact_results = np.full(
            (self._n_chans, self._n_chans, results.shape[1], results.shape[2]),
            fill_value=fill,
        )
        for con_result, seed, target in zip(
            results, self._seeds, self._targets
        ):
            compact_results[seed, target] = con_result

        # remove empty rows and cols
        filled_rows = []
        for row_i, row in enumerate(compact_results):
            if not np.all(row == fill):
                filled_rows.append(row_i)
        filled_cols = []
        for col_i, col in enumerate(compact_results.transpose(1, 0, 2, 3)):
            if not np.all(col == fill):
                filled_cols.append(col_i)
        compact_results = compact_results[np.ix_(filled_rows, filled_cols)]

        indices = (
            np.unique(self._seeds),
            np.unique(self._targets),
        )

        return compact_results.copy(), indices

    def copy(self):
        """Return a copy of the object."""
        return copy.deepcopy(self)


def compute_fft(
    data: np.ndarray, sfreq: float, verbose: bool = True
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the FFT on real-valued data.

    As the data is assumed to be real-valued, only those values corresponding
    to the positive frequencies are returned.

    PARAMETERS
    ----------
    data : numpy ndarray
    -   Array of real-valued data to compute the FFT on, with shape [epochs x
        channels x times].

    sfreq : float
    -   Sampling frequency of the data in Hz.

    RETURNS
    -------
    fft : NumPy ndarray
    -   FFT of the data with shape [epochs x channels x positive frequencies].

    freqs : NumPy ndarray
    -   Frequencies of `fft`.

    verbose : bool; default True
    -   Whether or not to report the status of the computation.

    RAISES
    ------
    ValueError
    -   Raised if `data` is not a numpy ndarray or does not have 3 dimensions.
    """
    if not isinstance(data, np.ndarray) or data.ndim != 3:
        raise ValueError("`data` must be a 3D NumPy array.")

    if not np.isreal(data).all():
        warn("`data` is expected to be real-valued.\n", UserWarning)

    if verbose:
        print("Computing FFT on the data...")

    freqs = np.linspace(0, sfreq / 2, sfreq + 1)
    fft = np.fft.fft(sp.signal.detrend(data) * np.hanning(data.shape[2]))

    if verbose:
        print("    [FFT computation finished]\n")

    return fft[..., 1 : len(freqs)], freqs[1:]  # ignore zero freq.
