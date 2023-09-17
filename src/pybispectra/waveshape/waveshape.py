"""Tools for handling wave shape analysis."""

from copy import deepcopy

import numpy as np
from pqdm.processes import pqdm

from pybispectra.utils.results import ResultsWaveShape
from pybispectra.utils._process import (
    _ProcessBispectrum,
    _compute_bispectrum,
    _compute_threenorm,
)


class WaveShape(_ProcessBispectrum):
    """Class for computing wave shape properties using bicoherence.

    Parameters
    ----------
    data : numpy.ndarray, shape of [epochs, channels, frequencies]
        Fourier coefficients.

    freqs : numpy.ndarray, shape of [frequencies]
        Frequencies in :attr:`data`.

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
        Frequencies in :attr:`data`.

    sampling_freq : int | float
        Sampling frequency (in Hz) of the data from which :attr:`data` was
        derived.

    verbose : bool
        Whether or not to report the progress of the processing.

    Notes
    -----
    It is recommended that spatio-spectral filtering for a given frequency band
    of interest has been performed on :attr:`data` before analysing wave shape
    properties :footcite:`Bartz2019`. Spatio-spectral filtering is recommended
    as it can enhance the signal-to-noise ratio of your data as well as
    mitigate the risks of source-mixing in the sensor space compromising the
    bicoherence patterns of the data :footcite:`Bartz2019`. Filtering can be
    performed with :class:`~pybispectra.utils.SpatioSpectralFilter`.

    References
    ----------
    .. footbibliography::
    """

    _bicoherence = None

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
            Indices of the channels to compute bicoherence within. If
            :obj:`None`, bicoherence within all channels is computed.

        f1s : tuple of int or float | None (default None), length of 2
            Start and end lower frequencies to compute bicoherence for,
            respectively. If :obj:`None`, all frequencies are used.

        f2s : tuple of int or float | None (default None), length of 2
            Start and end higher frequencies to compute bicoherence for,
            respectively. If :obj:`None`, all frequencies are used.

        n_jobs : int (default ``1``)
            The number of jobs to run in parallel. If ``-1``, all available
            CPUs are used.

        Notes
        -----
        Bicoherence, :math:`\boldsymbol{\mathcal{B}}`, is the normalised
        version of the bispectrum, :math:`\textbf{B}`, which has the general
        form

        :math:`\textbf{B}_{kmn}(f_1,f_2)=<\textbf{k}(f_1)\textbf{m}(f_2)
        \textbf{n}^*(f_2+f_1)>`,

        where :math:`kmn` corresponds to the channels in the data, and the
        angled brackets represent the averaged value over epochs. For the
        purposes of waveshape analyses, bicoherence is only computed within a
        single signal, :math:`\textbf{x}`, such that

        :math:`\textbf{B}_{kmn}(f_1,f_2) := \textbf{B}_{xxx}(f_1,f_2)`.

        Normalisation of the bispectrum to bicoherence is achieved with the
        threenorm, :math:`\textbf{N}` :footcite:`Zandvoort2021`,

        :math:`\textbf{N}_{xxx}(f_1,f_2)=(<|\textbf{x}(f_1)|^3>
        <|\textbf{x}(f_2)|^3><|\textbf{x}(f_2+f_1)|^3>)^{\frac{1}{3}}`,

        :math:`\boldsymbol{\mathcal{B}}_{xxx}(f_1,f_2)=\Large
        \frac{\textbf{B}_{xxx}(f_1,f_2)}{\textbf{N}_{xxx}(f_1,f_2)}`.

        The threenorm is a form of univariate normalisation, whereby the values
        of the bicoherence will be bound in the range :math:`[0, 1]` in a
        manner that is independent of the coupling properties within or between
        signals :footcite:`Shahbazi2014`.

        Bicoherence is computed for all values of :attr:`f1s` and :attr:`f2s`.
        If any value of :attr:`f1s` is higher than :attr:`f2s`, a
        :obj:`numpy.nan` value is returned.

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

        bispectrum = self._compute_bispectrum()
        threenorm = self._compute_threenorm()
        self._bicoherence = bispectrum / threenorm
        self._store_results()

        if self.verbose:
            print("    ... Bicoherence computation finished\n")

    def _reset_attrs(self) -> None:
        """Reset attrs. of the object to prevent interference."""
        super()._reset_attrs()

        self._bicoherence = None

    def _sort_indices(self, indices: tuple[int]) -> None:
        """Sort channel indices inputs."""
        indices = deepcopy(indices)
        if indices is None:
            indices = tuple(range(self._n_chans))
        if not isinstance(indices, tuple):
            raise TypeError("`indices` must be a tuple.")
        if any(not isinstance(idx, (int, np.integer)) for idx in indices):
            raise TypeError("Entries of `indices` must be ints.")

        if any(idx < 0 or idx >= self._n_chans for idx in indices):
            raise ValueError(
                "`indices` contains indices for channels not present in "
                "the data."
            )

        self._n_cons = len(indices)
        self._indices = indices

    def _compute_bispectrum(self) -> None:
        """Compute bispectrum between f1s and f2s within channels.

        Returns
        -------
        bispectrum : np.ndarray, shape of [channels, f1s, f2s]
            Complex-valued array containing the bispectrum for each channel.
        """
        if self.verbose:
            print("    Computing bispectrum...")

        args = [
            {
                "data": self.data[:, channel][:, None],
                "freqs": self.freqs,
                "f1s": self._f1s,
                "f2s": self._f2s,
                "kmn": np.array([np.array([0, 0, 0])]),
            }
            for channel in self._indices
        ]

        # have to average complex value outside of Numba-compiled function
        bispectrum = (
            np.array(
                pqdm(
                    args,
                    _compute_bispectrum,
                    self._n_jobs,
                    argument_type="kwargs",
                    desc="Processing connections...",
                    disable=not self.verbose,
                )
            )
            .mean(axis=2)
            .transpose(1, 0, 2, 3)
        )[0]

        if self.verbose:
            print("        ... Bispectrum computation finished\n")

        return bispectrum

    def _compute_threenorm(self) -> None:
        """Compute threenorm between f1s and f2s within channels.

        Returns
        -------
        threenorm : numpy.ndarray, shape of [channels, f1s, f2s]
            Complex-valued array containing the threenorm for each channel.
        """
        if self.verbose:
            print("    Computing threenorm...")

        args = [
            {
                "data": self.data[:, channel][:, None],
                "freqs": self.freqs,
                "f1s": self._f1s,
                "f2s": self._f2s,
                "kmn": np.array([np.array([0, 0, 0])]),
            }
            for channel in self._indices
        ]

        threenorm = np.array(
            pqdm(
                args,
                _compute_threenorm,
                self._n_jobs,
                argument_type="kwargs",
                desc="Processing connections...",
                disable=not self.verbose,
            )
        ).transpose(1, 0, 2, 3)[0]

        if self.verbose:
            print("        ... Threenorm computation finished\n")

        return threenorm

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
            The results of the wave shape computation.
        """
        return deepcopy(self._results)
