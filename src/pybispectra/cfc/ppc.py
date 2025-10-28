"""Tools for handling PPC analysis."""

import numpy as np
from numba import njit

from pybispectra.utils import ResultsCFC
from pybispectra.utils._defaults import _precision
from pybispectra.utils._process import _ProcessFreqBase
from pybispectra.utils._utils import _fast_find_first, _compute_in_parallel


class PPC(_ProcessFreqBase):
    """Class for computing phase-phase coupling (PPC).

    Parameters
    ----------
    data : ~numpy.ndarray of float, shape of [epochs, channels, frequencies (, times)]
        Fourier coefficients.

    freqs : ~numpy.ndarray of float, shape of [frequencies]
        Frequencies (in Hz) in ``data``. Frequencies are expected to be evenly
        spaced.

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
        Compute PPC, averaged over epochs.

    copy :
        Return a copy of the object.

    Attributes
    ----------
    results : ~pybispectra.utils.ResultsCFC
        PPC results.

    data : ~numpy.ndarray of float, shape of [epochs, channels, frequencies (, times)]
        Fourier coefficients.

    freqs : ~numpy.ndarray of float, shape of [frequencies]
        Frequencies (in Hz) in ``data``.

    sampling_freq : int | float
        Sampling frequency (in Hz) of the data from which ``data`` was derived.

    times : ~numpy.ndarray, shape of [times] | None
        Timepoints (in seconds) in ``data``.

    verbose : bool
        Whether or not to report the progress of the processing.
    """

    _ppc: np.ndarray = None

    def compute(
        self,
        indices: tuple[tuple[int]] | None = None,
        f1s: tuple[int | float] | None = None,
        f2s: tuple[int | float] | None = None,
        times: tuple[int | float] | None = None,
        n_jobs: int = 1,
    ) -> None:
        r"""Compute PPC, averaged over epochs.

        Parameters
        ----------
        indices : tuple of tuple of int, length of 2 | None (default None)
            Indices of the seed and target channels, respectively, to compute PPC
            between. If :obj:`None`, coupling between all channels is computed.

        f1s : tuple of int or float, length of 2 | None (default None)
            Start and end lower frequencies to compute PPC on, respectively. If
            :obj:`None`, all frequencies are used.

        f2s : tuple of int or float, length of 2 | None (default None)
            Start and end higher frequencies to compute PPC on, respectively.
            If :obj:`None`, all frequencies are used.

        times : tuple of int or float, length of 2 | None (default None)
            Start and end times (in seconds) to compute PPC on, respectively. If
            :obj:`None`, all timepoints are used.

            .. versionadded:: 1.3

        n_jobs : int (default ``1``)
            Number of jobs to run in parallel. If ``-1``, all available CPUs are used.

        Notes
        -----
        PPC is computed as coherence between frequencies :footcite:`Giehl2021`

        :math:`\textrm{PPC}(\textbf{x}_{f_1},\textbf{y}_{f_2})=\Large \frac{|\langle
        \textbf{a}_x(f_1)\textbf{a}_y(f_2) e^{i(\boldsymbol{\varphi}_x(f_1)\frac{f_2}
        {f_1}-\boldsymbol{\varphi}_y(f_2))} \rangle|}{\langle\textbf{a}_x(f_1)
        \textbf{a}_y(f_2) \rangle}` ,

        where :math:`\textbf{a}(f)` and :math:`\boldsymbol{\varphi}(f)` are the
        amplitude and phase of a signal at a given frequency, respectively; :math:`f_1`
        and :math:`f_2` correspond to a lower and higher frequency, respectively; and
        :math:`<>` represents the average value over epochs.

        PPC is computed between all values of ``f1s`` and ``f2s``.

        .. warning::
            For values of ``f1s`` higher than ``f2s`` or where ``f2s + f1s`` exceeds the
            Nyquist frequency, a :obj:`numpy.nan` value is returned.

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
            print("Computing PPC...")

        self._compute_ppc()
        self._store_results()

        if self.verbose:
            print("    ... PPC computation finished\n")

    def _reset_attrs(self) -> None:
        """Reset attrs. of the object to prevent interference."""
        super()._reset_attrs()
        self._ppc = None

    def _compute_ppc(self) -> None:
        """Compute PPC between f1s of seeds and f2s of targets."""
        loop_kwargs = [
            {"data": self.data[:, (seed, target)][..., self._time_idcs]}
            for seed, target in zip(self._seeds, self._targets)
        ]
        static_kwargs = {
            "freqs": self.freqs,
            "f1s": self._f1s,
            "f2s": self._f2s,
            "precision": _precision.real,
        }
        try:
            self._ppc = _compute_in_parallel(
                func=_compute_ppc,
                loop_kwargs=loop_kwargs,
                static_kwargs=static_kwargs,
                output=np.zeros(
                    (self._n_cons, self._f1s.size, self._f2s.size, self._times.size),
                    dtype=_precision.real,
                ),
                message="Processing connections...",
                n_jobs=self._n_jobs,
                verbose=self.verbose,
                prefer="processes",
            )
        except MemoryError as error:  # pragma: no cover
            raise MemoryError(
                "Memory allocation for the PPC computation failed. Try reducing the "
                "sampling frequency of the data, or reduce the precision of the "
                "computation with `pybispectra.set_precision('single')`."
            ) from error

        if self.times is None:  # remove placeholder time dimension
            self._ppc = self._ppc[..., 0]

    def _store_results(self) -> None:
        """Store computed results in an object."""
        self._results = ResultsCFC(
            data=self._ppc,
            indices=self._indices,
            f1s=self._f1s,
            f2s=self._f2s,
            times=self._times,
            name="PPC",
        )

    @property
    def results(self) -> ResultsCFC:
        return self._results


@njit
def _compute_ppc(
    data: np.ndarray,
    freqs: np.ndarray,
    f1s: np.ndarray,
    f2s: np.ndarray,
    precision: type,
) -> np.ndarray:  # pragma: no cover
    """Compute PPC for a single connection across epochs.

    Parameters
    ----------
    data : numpy.ndarray, shape of [epochs, 2, frequencies]
        FFT coefficients where the second dimension contains the data for the seed and
        target channel of a single connection, respectively.

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
        PPC for a single connection.
    """
    results = np.full((f1s.shape[0], f2s.shape[0]), fill_value=np.nan, dtype=precision)
    f1_start = _fast_find_first(freqs, f1s[0], 0)
    f1_end = _fast_find_first(freqs, f1s[-1], f1_start)
    f2_start = _fast_find_first(freqs, f2s[0], 0)
    f2_end = _fast_find_first(freqs, f2s[-1], f2_start)
    for f1_ri, f1_fi in enumerate(range(f1_start, f1_end + 1)):
        f1 = freqs[f1_fi]
        for f2_ri, f2_fi in enumerate(range(f2_start, f2_end + 1)):
            f2 = freqs[f2_fi]
            if f1 < f2 and f1 > 0:
                fft_f1 = data[:, 0, f1_fi]
                fft_f2 = data[:, 1, f2_fi]
                numerator = np.abs(
                    (
                        np.abs(fft_f1)
                        * np.abs(fft_f2)
                        * np.exp(
                            1j
                            * (
                                np.angle(fft_f1, True) * (f2 / f1)
                                - np.angle(fft_f2, True)
                            )
                        )
                    ).mean()
                )
                denominator = np.mean((np.abs(fft_f1) * np.abs(fft_f2)))
                results[f1_ri, f2_ri] = numerator / denominator

    return results
