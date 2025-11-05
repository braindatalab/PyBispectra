"""Tools for handling AAC analysis."""

import numpy as np
from numba import njit

from pybispectra.utils import ResultsCFC
from pybispectra.utils._defaults import _precision
from pybispectra.utils._process import _ProcessFreqBase
from pybispectra.utils._utils import (
    _compute_pearsonr_2d,
    _fast_find_first,
    _compute_in_parallel,
)


class AAC(_ProcessFreqBase):
    """Class for computing amplitude-amplitude coupling (AAC).

    Parameters
    ----------
    data : ~numpy.ndarray, shape of [epochs, channels, frequencies, times]
        Amplitude (power) of the time-frequency representation (TFR) of data.

    freqs : ~numpy.ndarray, shape of [frequencies]
        Frequencies (in Hz) in ``data``. Frequencies are expected to be evenly spaced.

    sampling_freq : int | float
        Sampling frequency (in Hz) of the data from which ``data`` was derived.

    times : ~numpy.ndarray, shape of [times] | None
        Timepoints (in seconds) in ``data``. If ``data`` has a times dimension and
        ``times = None``, the time of the first sample in ``data`` is assumed to be 0
        seconds.

        .. versionadded:: 1.3

    verbose : bool (default True)
        Whether or not to report the progress of the processing.

    Methods
    -------
    compute :
        Compute AAC, averaged over epochs.

    copy :
        Return a copy of the object.

    Attributes
    ----------
    results : ~pybispectra.utils.ResultsCFC
        AAC results.

    data : ~numpy.ndarray, shape of [epochs, channels, frequencies, times]
        Amplitude (power) of the TFR of data.

    freqs : ~numpy.ndarray, shape of [frequencies]
        Frequencies (in Hz) in ``data``.

    sampling_freq : int | float
        Sampling frequency (in Hz) of ``data``.

    verbose : bool
        Whether or not to report the progress of the processing.
    """  # noqa: E501

    _data_precision: type = _precision.real  # TFR real-valued

    _data_ndims: int = (4,)  # [epochs, channels, frequencies, times]

    _aac: np.ndarray = None

    def compute(
        self,
        indices: tuple[tuple[int]] | None = None,
        f1s: tuple[int | float] | None = None,
        f2s: tuple[int | float] | None = None,
        times: tuple[int | float] | None = None,
        n_jobs: int = 1,
    ) -> None:
        r"""Compute AAC, averaged over epochs.

        Parameters
        ----------
        indices : tuple of tuple of int, length of 2 | None (default None)
            Indices of the seed and target channels, respectively, to compute AAC
            between. If :obj:`None`, coupling between all channels is computed.

        f1s : tuple of int or float, length of 2 | None (default None)
            Start and end lower frequencies to compute AAC on, respectively. If
            :obj:`None`, all frequencies are used.

        f2s : tuple of int or float, length of 2 | None (default None)
            Start and end higher frequencies to compute AAC on, respectively. If
            :obj:`None`, all frequencies are used.

        times : tuple of int or float, length of 2 | None (default None)
            Start and end times (in seconds) to compute AAC on, respectively. If
            :obj:`None`, all timepoints are used.

            .. versionadded:: 1.3

        n_jobs : int (default ``1``)
            Number of jobs to run in parallel. If ``-1``, all available CPUs are used.

        Notes
        -----
        AAC is computed as the Pearson correlation coefficient across times for each
        frequency in each epoch, with coupling being averaged across epochs
        :footcite:`Giehl2021`.

        AAC is computed between all values of ``f1s`` and ``f2s``.

        .. warning::
            For values of ``f1s`` higher than ``f2s``, a :obj:`numpy.nan` value is
            returned.

        References
        ----------
        .. footbibliography::
        """
        self._reset_attrs()

        self._sort_indices(indices)
        self._sort_freqs(f1s, f2s)
        self._sort_tmin_tmax(times)
        self._sort_parallelisation(n_jobs)

        if self.verbose:
            print("Computing AAC...")

        self._compute_aac()
        self._store_results()

        if self.verbose:
            print("    ... AAC computation finished\n")

    def _reset_attrs(self) -> None:
        """Reset attrs. of the object to prevent interference."""
        super()._reset_attrs()
        self._aac = None

    def _compute_aac(self) -> None:
        """Compute AAC between f1s of seeds and f2s of targets."""
        loop_kwargs = [
            {"data": self._data[:, (seed, target)][..., self._time_idcs]}
            for seed, target in zip(self._seeds, self._targets)
        ]
        static_kwargs = {
            "freqs": self.freqs,
            "f1s": self._f1s,
            "f2s": self._f2s,
            "precision": _precision.real,
        }
        try:
            self._aac = _compute_in_parallel(
                func=_compute_aac,
                loop_kwargs=loop_kwargs,
                static_kwargs=static_kwargs,
                output=np.zeros(
                    (self._n_cons, self._f1s.size, self._f2s.size),
                    dtype=_precision.real,
                ),
                message="Processing connections...",
                n_jobs=self._n_jobs,
                verbose=self.verbose,
                prefer="processes",
            )
        except MemoryError as error:  # pragma: no cover
            raise MemoryError(
                "Memory allocation for the AAC computation failed. Try reducing the "
                "sampling frequency of the data, or reduce the precision of the "
                "computation with `pybispectra.set_precision('single')`."
            ) from error

    def _store_results(self) -> None:
        """Store computed results in an object."""
        self._results = ResultsCFC(
            data=self._aac,
            indices=self._indices,
            f1s=self._f1s,
            f2s=self._f2s,
            name="AAC",
        )

    @property
    def results(self) -> ResultsCFC:
        return self._results


@njit
def _compute_aac(
    data: np.ndarray,
    freqs: np.ndarray,
    f1s: np.ndarray,
    f2s: np.ndarray,
    precision: type,
) -> np.ndarray:  # pragma: no cover
    """Compute AAC for a single connection across epochs.

    Parameters
    ----------
    data : numpy.ndarray, shape of [epochs, 2, frequencies, times]
        Amplitude (power) of the TFR of data where the second dimension contains the
        data for the seed and target channel of a single connection, respectively.

    freqs : numpy.ndarray, shape of [frequencies]
        Frequencies in ``data``.

    f1s : numpy.ndarray, shape of [low frequencies]
        Low frequencies to compute coupling for.

    f2s : numpy.ndarray, shape of [high frequencies]
        High frequencies to compute coupling for.

    precision : type
        Precision to use for the computation. Either ``numpy.float32`` (single) or
        ``numpy.float64`` (double).

    Returns
    -------
    results : numpy.ndarray, shape of [low frequencies, high frequencies]
        AAC averaged across epochs for a single connection.
    """
    results = np.full((f1s.size, f2s.size), fill_value=np.nan, dtype=precision)
    f1_start = _fast_find_first(freqs, f1s[0], 0)
    f1_end = _fast_find_first(freqs, f1s[-1], f1_start)
    f2_start = _fast_find_first(freqs, f2s[0], 0)
    f2_end = _fast_find_first(freqs, f2s[-1], f2_start)
    for f1_ri, f1_fi in enumerate(range(f1_start, f1_end + 1)):
        f1 = freqs[f1_fi]
        for f2_ri, f2_fi in enumerate(range(f2_start, f2_end + 1)):
            f2 = freqs[f2_fi]
            if f1 <= f2 and f1 > 0:
                results[f1_ri, f2_ri] = np.mean(
                    _compute_pearsonr_2d(
                        data[:, 0, f1_fi], data[:, 1, f2_fi], precision
                    )
                )

    return results
