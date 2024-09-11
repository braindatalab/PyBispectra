"""Tools for handling waveshape analysis."""

from copy import deepcopy

import numpy as np

from pybispectra.utils._defaults import _precision
from pybispectra.utils._process import (
    _compute_bispectrum,
    _compute_threenorm,
    _ProcessBispectrum,
)
from pybispectra.utils.results import ResultsWaveShape
from pybispectra.utils._utils import _compute_in_parallel

np.seterr(divide="ignore", invalid="ignore")  # no warning for NaN division


class WaveShape(_ProcessBispectrum):
    """Class for computing waveshape properties using bicoherence.

    Parameters
    ----------
    data : numpy.ndarray, shape of [epochs, channels, frequencies]
        Fourier coefficients.

    freqs : numpy.ndarray, shape of [frequencies]
        Frequencies (in Hz) in :attr:`data`. Frequencies are expected to be evenly
        spaced.

    sampling_freq : int | float
        Sampling frequency (in Hz) of the data from which :attr:`data` was derived.

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
    results : tuple of ~pybispectra.utils.ResultsWaveShape
        Bicoherence of the data.

    data : ~numpy.ndarray, shape of [epochs, channels, frequencies]
        Fourier coefficients.

    freqs : ~numpy.ndarray, shape of [frequencies]
        Frequencies (in Hz) in :attr:`data`.

    sampling_freq : int | float
        Sampling frequency (in Hz) of the data from which :attr:`data` was derived.

    verbose : bool
        Whether or not to report the progress of the processing.

    Notes
    -----
    It is recommended that spatio-spectral filtering for a given frequency band of
    interest has been performed before analysing waveshape properties
    :footcite:`Bartz2019`. This can enhance the signal-to-noise ratio of your data as
    well as mitigate the risks of source-mixing in the sensor space compromising the
    bicoherence patterns of the data :footcite:`Bartz2019`. Filtering can be performed
    with :class:`~pybispectra.utils.SpatioSpectralFilter`.

    References
    ----------
    .. footbibliography::
    """

    def compute(
        self,
        indices: tuple[int] | None = None,
        f1s: tuple[int | float] | None = None,
        f2s: tuple[int | float] | None = None,
        n_jobs: int = 1,
    ) -> None:
        r"""Compute bicoherence within channels, averaged over epochs.

        Parameters
        ----------
        indices : tuple of int | None (default None)
            Indices of the channels to compute bicoherence within. If :obj:`None`,
            bicoherence within all channels is computed.

        f1s : tuple of int or float, length of 2 | None (default None)
            Start and end lower frequencies to compute bicoherence for, respectively. If
            :obj:`None`, all frequencies are used.

        f2s : tuple of int or float, length of 2 | None (default None)
            Start and end higher frequencies to compute bicoherence for, respectively.
            If :obj:`None`, all frequencies are used.

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

        Bicoherence is computed for all values of :attr:`f1s` and :attr:`f2s`. If any
        value of :attr:`f1s` is higher than :attr:`f2s`, a :obj:`numpy.nan` value is
        returned.

        References
        ----------
        .. footbibliography::
        """
        self._reset_attrs()

        self._sort_indices(indices)
        self._sort_freqs(f1s, f2s)
        self._sort_parallelisation(n_jobs)

        if self.verbose:
            print("Computing bicoherence...\n")

        self._compute_bispectrum()
        self._compute_threenorm()
        self._bicoherence = self._bispectrum / self._threenorm
        self._store_results()

        if self.verbose:
            print("    ... Bicoherence computation finished\n")

    def _reset_attrs(self) -> None:
        """Reset attrs. of the object to prevent interference."""
        super()._reset_attrs()

        self._bispectrum = None
        self._threenorm = None
        self._bicoherence = None

    def _sort_indices(self, indices: tuple[int]) -> None:
        """Sort channel indices inputs."""
        indices = deepcopy(indices)
        if indices is None:
            indices = tuple(range(self._n_chans))
        if not isinstance(indices, tuple):
            raise TypeError("`indices` must be a tuple.")
        if any(not isinstance(idx, int) for idx in indices):
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

        loop_kwargs = [{"data": self.data[:, [channel]]} for channel in self._indices]
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
                    (self._n_cons, 1, self._f1s.size, self._f2s.size),
                    dtype=_precision.complex,
                ),
                message="Processing connections...",
                n_jobs=self._n_jobs,
                verbose=self.verbose,
                prefer="processes",
            ).transpose(1, 0, 2, 3)[0]
        except MemoryError as error:  # pragma: no cover
            raise MemoryError(
                "Memory allocation for the bispectrum computation failed. Try reducing "
                "the sampling frequency of the data, or reduce the precision of the "
                "computation with `pybispectra.set_precision('single')`."
            ) from error

        if self.verbose:
            print("        ... Bispectrum computation finished\n")

    def _compute_threenorm(self) -> None:
        """Compute threenorm between f1s and f2s within channels."""
        if self.verbose:
            print("    Computing threenorm...")

        loop_kwargs = [{"data": self.data[:, [channel]]} for channel in self._indices]
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
                    (self._n_cons, 1, self._f1s.size, self._f2s.size),
                    dtype=_precision.real,
                ),
                message="Processing connections...",
                n_jobs=self._n_jobs,
                verbose=self.verbose,
                prefer="processes",
            ).transpose(1, 0, 2, 3)[0]
        except MemoryError as error:  # pragma: no cover
            raise MemoryError(
                "Memory allocation for the threenorm computation failed. Try reducing "
                "the sampling frequency of the data, or reduce the precision of the "
                "computation with `pybispectra.set_precision('single')`."
            ) from error

        if self.verbose:
            print("        ... Threenorm computation finished\n")

    def _store_results(self) -> None:
        """Store computed results in objects."""
        self._results = ResultsWaveShape(
            data=self._bicoherence,
            indices=self._indices,
            f1s=self._f1s,
            f2s=self._f2s,
            name="Waveshape",
        )

    @property
    def results(self) -> ResultsWaveShape:
        """Return the results.

        Returns
        -------
        results : ~pybispectra.utils.ResultsWaveShape
            The results of the waveshape computation.
        """
        return deepcopy(self._results)
