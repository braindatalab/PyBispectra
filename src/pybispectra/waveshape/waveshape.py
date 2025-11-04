"""Tools for handling waveshape analysis."""

import numpy as np

from pybispectra.utils._defaults import _precision
from pybispectra.utils._process import (
    _compute_bispectrum,
    _compute_threenorm,
    _ProcessBispectrum,
)
from pybispectra.utils.results import ResultsWaveShape
from pybispectra.utils._utils import _compute_in_parallel, _int_like

np.seterr(divide="ignore", invalid="ignore")  # no warning for NaN division


class WaveShape(_ProcessBispectrum):
    """Class for computing waveshape properties using the bispectrum.

    Parameters
    ----------
    data : ~numpy.ndarray, shape of [epochs, channels, frequencies (, times)]
        Fourier coefficients.

    freqs : ~numpy.ndarray, shape of [frequencies]
        Frequencies (in Hz) in ``data``. Frequencies are expected to be evenly spaced.

    sampling_freq : int | float
        Sampling frequency (in Hz) of the data from which ``data`` was derived.

    times : ~numpy.ndarray, shape of [times] | None
        Timepoints (in seconds) in ``data``. If ``data`` has a times dimension and
        ``times = None``, the time of the first sample in ``data`` is assumed to be 0
        seconds.

    verbose : bool (default True)
        Whether or not to report the progress of the processing.

    Methods
    -------
    compute :
        Compute bicoherence within channels, averaged over epochs.

    copy :
        Return a copy of the object.

    Attributes
    ----------
    results : ~pybispectra.utils.ResultsWaveShape | tuple of ~pybispectra.utils.ResultsWaveShape
        Waveshape results for each of the computed metrics.

    data : ~numpy.ndarray, shape of [epochs, channels, frequencies (, times)]
        Fourier coefficients.

    freqs : ~numpy.ndarray, shape of [frequencies]
        Frequencies (in Hz) in ``data``.

    sampling_freq : int | float
        Sampling frequency (in Hz) of the data from which ``data`` was derived.

    times : ~numpy.ndarray, shape of [times] | None
        Timepoints (in seconds) in ``data``.

    verbose : bool
        Whether or not to report the progress of the processing.

    Notes
    -----
    It is recommended that spatio-spectral filtering for a given frequency band of
    interest has been performed before analysing waveshape properties
    :footcite:`Bartz2019`. This can enhance the signal-to-noise ratio of your data as
    well as mitigate the risks of source-mixing in the sensor space compromising the
    bicoherence patterns of the data :footcite:`Bartz2019`. Filtering can be performed
    with :class:`pybispectra.utils.SpatioSpectralFilter`.

    References
    ----------
    .. footbibliography::
    """  # noqa: E501

    _return_nonorm = False
    _return_threenorm = False

    def compute(
        self,
        indices: tuple[int] | None = None,
        f1s: tuple[int | float] | None = None,
        f2s: tuple[int | float] | None = None,
        times: tuple[int | float] | None = None,
        norm: bool | tuple[bool] = True,
        n_jobs: int = 1,
    ) -> None:
        r"""Compute waveshape within channels, averaged over epochs.

        Parameters
        ----------
        indices : tuple of int | None (default None)
            Indices of the channels to compute waveshape within. If :obj:`None`,
            waveshape within all channels is computed.

        f1s : tuple of int or float, length of 2 | None (default None)
            Start and end lower frequencies to compute waveshape for, respectively. If
            :obj:`None`, all frequencies are used.

        f2s : tuple of int or float, length of 2 | None (default None)
            Start and end higher frequencies to compute waveshape for, respectively.
            If :obj:`None`, all frequencies are used.

        times : tuple of int or float, length of 2 | None (default None)
            Start and end times (in seconds) to compute waveshape for, respectively. If
            :obj:`None`, all timepoints are used.

            .. versionadded:: 1.3

        norm : bool | tuple of bool (default True)
            Whether to normalise the waveshape results using the threenorm. If a tuple
            of bool, both forms of waveshape are computed in turn.

            ..versionadded:: 1.3

        n_jobs : int (default ``1``)
            The number of jobs to run in parallel. If ``-1``, all available CPUs are
            used.

        Notes
        -----
        Non-sinudoisal waveshape features can be extracted using bispectrum-based
        methods. The bispectrum has the general form

        :math:`\textbf{B}_{kmn}(f_1,f_2)=<\textbf{k}(f_1)\textbf{m}(f_2)
        \textbf{n}^*(f_2+f_1)>` ,

        where :math:`kmn` is a combination of signals with Fourier coefficients
        :math:`\textbf{k}`, :math:`\textbf{m}`, and :math:`\textbf{n}`, respectively;
        :math:`f_1` and :math:`f_2` correspond to a lower and higher frequency,
        respectively; and :math:`<>` represents the average value over epochs. When
        analysing waveshape, we are interested in only a single signal, and as such
        :math:`k=m=n`.

        Furthermore, we can normalise the bispectrum to the bicoherence,
        :math:`\boldsymbol{\mathcal{B}}`, using the threenorm, :math:`\textbf{N}`,
        :footcite:`Shahbazi2014`

        :math:`\textbf{N}_{xxx}(f_1,f_2)=(<|\textbf{x}(f_1)|^3><|\textbf{x} (f_2)|^3>
        <|\textbf{x}(f_2+f_1)|^3>)^{\frac{1}{3}}` ,

        :math:`\boldsymbol{\mathcal{B}}_{xxx}(f_1,f_2)=\Large\frac{\textbf{B}_{xxx}
        (f_1,f_2)}{\textbf{N}_{xxx}(f_1,f_2)}` ,

        where the resulting values lie in the range :math:`[-1, 1]`.

        Waveshape is computed for all values of ``f1s`` and ``f2s``.

        .. warning::
            For values of ``f1s`` higher than ``f2s`` or where ``f2s + f1s`` exceeds the
            Nyquist frequency, a :obj:`numpy.nan` value is returned.

        References
        ----------
        .. footbibliography::
        """
        self._reset_attrs()

        self._sort_metrics(norm)
        self._sort_indices(indices)
        self._sort_freqs(f1s, f2s)
        self._sort_tmin_tmax(times)
        self._sort_parallelisation(n_jobs)

        if self.verbose:
            print("Computing waveshape...\n")

        self._compute_bispectrum()
        if self._return_threenorm:
            self._compute_threenorm()
            self._bicoherence = self._bispectrum / self._threenorm
        self._store_results()

        if self.verbose:
            print("    ... Waveshape computation finished\n")

    def _reset_attrs(self) -> None:
        """Reset attrs. of the object to prevent interference."""
        super()._reset_attrs()

        self._return_nonorm = False
        self._return_threenorm = False

        self._bispectrum = None
        self._threenorm = None
        self._bicoherence = None

    def _sort_metrics(self, norm: bool | tuple[bool]) -> None:
        """Sort inputs for the form of results being requested."""
        if not isinstance(norm, (bool, tuple)):
            raise TypeError("`norm` must be a bool or tuple of bools.")

        if isinstance(norm, bool):
            norm = (norm,)

        if any(not isinstance(entry, bool) for entry in norm):
            raise TypeError("Entries of `norm` must be bools.")

        if False in norm:
            self._return_nonorm = True
        if True in norm:
            self._return_threenorm = True

    def _sort_indices(self, indices: tuple[int]) -> None:
        """Sort channel indices inputs."""
        if indices is None:
            indices = tuple(range(self._n_chans))
        if not isinstance(indices, tuple):
            raise TypeError("`indices` must be a tuple.")
        if any(not isinstance(idx, _int_like) for idx in indices):
            raise TypeError("Entries of `indices` must be ints.")

        if any(idx < 0 or idx >= self._n_chans for idx in indices):
            raise ValueError(
                "`indices` contains indices for channels not present in the data."
            )

        self._n_cons = len(indices)
        self._indices = indices

    def _compute_bispectrum(self) -> None:
        """Compute bispectrum between f1s and f2s within channels."""
        if self.verbose:
            print("    Computing bispectrum...")

        loop_kwargs = [
            {"data": self._data[:, [channel]][..., self._time_idcs]}
            for channel in self._indices
        ]
        static_kwargs = {
            "freqs": self.freqs,
            "f1s": self._f1s,
            "f2s": self._f2s,
            "kmn": np.array([np.array([0, 0, 0])]),
            "precision": _precision.complex,
        }
        try:
            self._bispectrum = _compute_in_parallel(
                func=_compute_bispectrum,
                loop_kwargs=loop_kwargs,
                static_kwargs=static_kwargs,
                output=np.zeros(
                    (self._n_cons, 1, self._f1s.size, self._f2s.size, self._times.size),
                    dtype=_precision.complex,
                ),
                message="Processing channels...",
                n_jobs=self._n_jobs,
                verbose=self.verbose,
                prefer="processes",
            ).transpose(1, 0, 2, 3, 4)[0]
        except MemoryError as error:  # pragma: no cover
            raise MemoryError(
                "Memory allocation for the bispectrum computation failed. Try reducing "
                "the sampling frequency of the data, or reduce the precision of the "
                "computation with `pybispectra.set_precision('single')`."
            ) from error

        if self.times is None:  # remove placeholder time dimension
            self._bispectrum = self._bispectrum[..., 0]

        if self.verbose:
            print("        ... Bispectrum computation finished\n")

    def _compute_threenorm(self) -> None:
        """Compute threenorm between f1s and f2s within channels."""
        if self.verbose:
            print("    Computing threenorm...")

        loop_kwargs = [
            {"data": self._data[:, [channel]][..., self._time_idcs]}
            for channel in self._indices
        ]
        static_kwargs = {
            "freqs": self.freqs,
            "f1s": self._f1s,
            "f2s": self._f2s,
            "kmn": np.array([np.array([0, 0, 0])]),
            "precision": _precision.real,
        }
        try:
            self._threenorm = _compute_in_parallel(
                func=_compute_threenorm,
                loop_kwargs=loop_kwargs,
                static_kwargs=static_kwargs,
                output=np.zeros(
                    (self._n_cons, 1, self._f1s.size, self._f2s.size, self._times.size),
                    dtype=_precision.real,
                ),
                message="Processing channels...",
                n_jobs=self._n_jobs,
                verbose=self.verbose,
                prefer="processes",
            ).transpose(1, 0, 2, 3, 4)[0]
        except MemoryError as error:  # pragma: no cover
            raise MemoryError(
                "Memory allocation for the threenorm computation failed. Try reducing "
                "the sampling frequency of the data, or reduce the precision of the "
                "computation with `pybispectra.set_precision('single')`."
            ) from error

        if self.times is None:  # remove placeholder time dimension
            self._threenorm = self._threenorm[..., 0]

        if self.verbose:
            print("        ... Threenorm computation finished\n")

    def _store_results(self) -> None:
        """Store computed results in objects."""
        results = []

        if self._return_nonorm:
            results.append(
                ResultsWaveShape(
                    data=self._bispectrum,
                    indices=self._indices,
                    f1s=self._f1s,
                    f2s=self._f2s,
                    times=self._times,
                    name="Waveshape | Bispectrum",
                )
            )

        if self._return_threenorm:
            results.append(
                ResultsWaveShape(
                    data=self._bicoherence,
                    indices=self._indices,
                    f1s=self._f1s,
                    f2s=self._f2s,
                    times=self._times,
                    name="Waveshape | Bicoherence",
                )
            )

        self._results = tuple(results)

    @property
    def results(self) -> ResultsWaveShape | tuple[ResultsWaveShape]:
        if len(self._results) == 1:
            return self._results[0]
        return self._results
