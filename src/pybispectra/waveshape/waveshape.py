"""Tools for handling wave shape analysis."""

import copy

import numpy as np
from pqdm.processes import pqdm

from pybispectra.utils._process import (
    _ProcessBispectrum,
    _compute_bispectrum,
    _compute_threenorm,
)


class WaveShape(_ProcessBispectrum):
    """Class for computing wave shape properties using bicoherence.

    Parameters
    ----------
    data : numpy.ndarray of float, shape [epochs x channels x frequencies]
        FFT coefficients.

    freqs : numpy.ndarray of float, shape [frequencies]
        Frequencies in :attr:`data`.

    verbose : bool (default True)
        Whether or not to report the progress of the processing.

    Attributes
    ----------
    results : tuple of ResultsWaveShape, shape [channels x f1 x f2]
        Bicoherence of the data.

    data : numpy.ndarray of float, shape [epochs x channels x frequencies]
        FFT coefficients.

    freqs : numpy.ndarray of float, shape [frequencies]
        Frequencies in :attr:`data`.

    indices : tuple of numpy.ndarray of int
        1D array of channel indices most recently used with :meth:`compute`.

    f1 : numpy.ndarray of float
        1D array of low frequencies most recently used with :meth:`compute`.

    f2 : numpy.ndarray of float
        1D array of high frequencies most recently used with :meth:`compute`.

    verbose : bool
        Whether or not to report the progress of the processing.

    Notes
    -----
    It is recommended that spatiospectral filtering for a given frequency band
    of interest has been performed on :attr:`data` before analysing wave shape
    properties :footcite:`Bartz2019`. This can be performed in PyBispectra
    using the SpatioSpectralFilter class (see the corresponding example file
    for details).

    References
    ----------
    .. footbibliography::
    """

    _bicoherence = None

    def compute(
        self,
        indices: np.ndarray | None = None,
        f1: np.ndarray | None = None,
        f2: np.ndarray | None = None,
        n_jobs: int = 1,
    ) -> None:
        r"""Compute bicoherence within channels, averaged over epochs.

        Parameters
        ----------
        indices : numpy.ndarray of int | None (default None)
            Indices of the channels to compute bicoherence within. If ``None``,
            bicoherence within all channels is computed.

        f1 : numpy.ndarray of float | None (default None)
            A 1D array of the lower frequencies to compute bicoherence for. If
            ``None``, all frequencies are used.

        f2 : numpy.ndarray of float | None (default None)
            A 1D array of the higher frequencies to compute bicoherence for. If
            ``None``, all frequencies are used.

        n_jobs : int (default ``1``)
            The number of jobs to run in parallel.

        Notes
        -----
        Bicoherence, :math:`\mathcal{B}`, is the normalised version of the
        bispectrum, :math:`B`, which has the general form

        :math:`\large B_{kmn}(f_1,f_2)=<\vec{k}(f_1)\vec{m}(f_2)\vec{n}^*(f_2+f_1)>`,

        where :math:`kmn` corresponds to the channels in the data, and the
        angled brackets represent the averaged value over epochs. For the
        purposes of waveshape analyses, bicoherence is only computed within a
        single signal, :math:`\vec{x}`, such that

        :math:`B_{kmn}(f_1,f_2) := B_{xxx}(f_1,f_2)`.

        Normalisation of the bispectrum to bicoherence is achieved with the
        threenorm, :math:`N` :footcite:`Zandvoort2021`,

        :math:`\large N_{xxx}(f_1,f_2)=(<|\vec{x}(f_1)|^3><|\vec{x}(f_2)|^3><|\vec{x}(f_2+f_1)|^3>)^{\frac{1}{3}}`,

        :math:`\large \mathcal{B}_{xxx}(f_1,f_2)=\Large \frac{B_{xxx}(f_1,f_2)}{N_{xxx}(f_1,f_2)}`.

        Bicoherence is computed for all values of :attr:`f1` and :attr:`f2`. If
        any value of :attr:`f1` is higher than :attr:`f2`, a ``numpy.nan``
        value is returned.

        References
        ----------
        .. footbibliography::
        """  # noqa E501
        self._reset_attrs()

        self._sort_indices(indices)
        self._sort_freqs(f1, f2)
        self._sort_parallelisation(n_jobs)

        if self.verbose:
            print("Computing bicoherence...\n")

        bispectrum = self._compute_bispectrum()
        threenorm = self._compute_threenorm()
        self._bicoherence = bispectrum / threenorm
        self._store_results()

        if self.verbose:
            print("    [Bicoherence computation finished]\n")

    def _reset_attrs(self) -> None:
        """Reset attrs. of the object to prevent interference."""
        super()._reset_attrs()

        self._bicoherence = None

    def _sort_indices(self, indices: tuple) -> None:
        """Sort channel indices inputs."""
        indices = copy.deepcopy(indices)
        if indices is None:
            indices = tuple(np.arange(self._n_chans))
        if not isinstance(indices, tuple):
            raise TypeError("`indices` should be a tuple.")

        if any(idx < 0 or idx >= self._n_chans for idx in indices):
            raise ValueError(
                "`indices` contains indices for channels not present in "
                "the data."
            )

        self._n_cons = len(indices)
        self.indices = indices

    def _compute_bispectrum(self) -> None:
        """Compute bispectrum between f1s and f2s within channels.

        Returns
        -------
        bispectrum : np.ndarray, shape [channels x f1 x f2]
            Complex-valued array containing the bispectrum for each channel.
        """
        if self.verbose:
            print("    Computing bispectrum...")

        args = [
            {
                "data": self.data[:, channel],
                "freqs": self.freqs,
                "f1s": self.f1,
                "f2s": self.f2,
                "kmn": np.array([np.array([0, 0, 0])]),
            }
            for channel in self.indices
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
        )

        if self.verbose:
            print("        [Bispectra computation finished]\n")

        return bispectrum

    def _compute_threenorm(self) -> None:
        """Compute threenorm between f1s and f2s within channels.

        Returns
        -------
        threenorm : np.ndarray, shape [channels x f1 x f2]
            Complex-valued array containing the threenorm for each channel.
        """
        if self.verbose:
            print("    Computing threenorm...")

        args = [
            {
                "data": self.data[:, channel],
                "freqs": self.freqs,
                "f1s": self.f1,
                "f2s": self.f2,
            }
            for channel in self.indices
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
        )

        if self.verbose:
            print("        [Threenorm computation finished]\n")

        return threenorm

    def _store_results(self) -> None:
        """"""
