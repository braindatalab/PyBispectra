"""Tools for handling general bispectrum and threenorm computations."""

import numpy as np

from pybispectra.utils import ResultsGeneral
from pybispectra.utils._defaults import _precision
from pybispectra.utils._process import (
    _compute_bispectrum,
    _compute_threenorm,
    _ProcessBispectrum,
)
from pybispectra.utils._utils import _compute_in_parallel


class _General(_ProcessBispectrum):
    """Base class for processing the bispectrum and threenorm."""

    _k: tuple[int] = None
    _m: tuple[int] = None
    _n: tuple[int] = None

    def _sort_indices(self, indices: tuple[tuple[int]] | None) -> None:
        """Sort kmn channel indices inputs."""
        if indices is None:
            indices = tuple(
                [
                    tuple(np.tile(range(self._n_chans), self._n_chans**2).tolist()),
                    tuple(
                        np.repeat(
                            np.tile(range(self._n_chans), self._n_chans), self._n_chans
                        ).tolist()
                    ),
                    tuple(np.repeat(range(self._n_chans), self._n_chans**2).tolist()),
                ]
            )
        if not isinstance(indices, tuple):
            raise TypeError("`indices` must be a tuple.")
        if len(indices) != 3:
            raise ValueError("`indices` must have length of 3.")
        self._indices = indices

        for group_idcs in indices:
            if not isinstance(group_idcs, tuple):
                raise TypeError("Entries of `indices` must be tuples.")
            if any(not isinstance(idx, int) for idx in group_idcs):
                raise TypeError("Entries for groups in `indices` must be ints.")
            if any(idx < 0 or idx >= self._n_chans for idx in group_idcs):
                raise ValueError(
                    "`indices` contains indices for channels not present in the data."
                )
        if len(np.unique([len(group) for group in indices])) != 1:
            raise ValueError("Entries of `indices` must have equal length.")

        self._k = self._indices[0]
        self._m = self._indices[1]
        self._n = self._indices[2]
        self._n_cons = len(self._indices[0])


class Bispectrum(_General):
    """Class for computing the bispectrum.

    Parameters
    ----------
    data : ~numpy.ndarray, shape of [epochs, channels, frequencies]
        Fourier coefficients.

    freqs : ~numpy.ndarray, shape of [frequencies]
        Frequencies (in Hz) in :attr:`data`. Frequencies are expected to be evenly
        spaced.

    sampling_freq : int | float
        Sampling frequency (in Hz) of the data from which :attr:`data` was derived.

    verbose : bool (default True)
        Whether or not to report the progress of the processing.

    Methods
    -------
    compute :
        Compute the bispectrum, averaged over epochs.

    copy :
        Return a copy of the object.

    Attributes
    ----------
    results : ~pybispectra.utils.ResultsGeneral
        Bispectrum results.

    data : ~numpy.ndarray of float, shape of [epochs, channels, frequencies]
        Fourier coefficients.

    freqs : ~numpy.ndarray of float, shape of [frequencies]
        Frequencies (in Hz) in :attr:`data`.

    sampling_freq : int | float
        Sampling frequency (in Hz) of the data from which :attr:`data` was derived.

    verbose : bool
        Whether or not to report the progress of the processing.

    Notes
    -----

    .. versionadded:: 1.2
    """

    def compute(
        self,
        indices: tuple[tuple[int]] | None = None,
        f1s: tuple[int | float] | None = None,
        f2s: tuple[int | float] | None = None,
        n_jobs: int = 1,
    ) -> None:
        r"""Compute the bispectrum, averaged over epochs.

        Parameters
        ----------
        indices : tuple of tuple of int, length of 3 | None (default None)
            Indices of the channels :math:`k`, :math:`m`, and :math:`n`, respectively,
            to compute the bispectrum for. If :obj:`None`, the bispectrum for all
            channel combinations is computed.

        f1s : tuple of int or float, length of 2 | None (default None)
            Start and end lower frequencies to compute the bispectrum for, respectively.
            If :obj:`None`, all frequencies are used.

        f2s : tuple of int or float, length of 2 | None (default None)
            Start and end higher frequencies to compute the bispectrum for,
            respectively. If :obj:`None`, all frequencies are used.

        n_jobs : int (default ``1``)
            The number of jobs to run in parallel. If ``-1``, all available CPUs are
            used.

        Notes
        -----
        The bispectrum, :math:`\textbf{B}`, has the general form

        :math:`\textbf{B}_{kmn}(f_1,f_2)=<\textbf{k}(f_1)\textbf{m}(f_2)
        \textbf{n}^*(f_2+f_1)>` ,

        where :math:`kmn` is a combination of signals with Fourier coefficients
        :math:`\textbf{k}`, :math:`\textbf{m}`, and :math:`\textbf{n}`, respectively;
        :math:`f_1` and :math:`f_2` correspond to a lower and higher frequency,
        respectively; and :math:`<>` represents the average value over epochs.

        The bispectrum is computed between all values of :attr:`f1s` and :attr:`f2s`. If
        any value of :attr:`f1s` is higher than :attr:`f2s`, a :obj:`numpy.nan` value is
        returned.
        """
        self._reset_attrs()

        self._sort_indices(indices)
        self._sort_freqs(f1s, f2s)
        self._sort_parallelisation(n_jobs)

        if self.verbose:
            print("Computing bispectrum...\n")

        self._compute_bispectrum()
        self._store_results()

        if self.verbose:
            print("    ... Bispectrum computation finished\n")

    def _reset_attrs(self) -> None:
        """Reset attrs. of the object to prevent interference."""
        super()._reset_attrs()

        self._bispectrum = None

    def _compute_bispectrum(self) -> None:
        """Compute bispectrum between f1s and f2s of seeds and targets."""
        loop_kwargs = [
            {"kmn": np.array([np.array([k, m, n])])}
            for (k, m, n) in zip(self._k, self._m, self._n)
        ]
        static_kwargs = {
            "data": self.data,
            "freqs": self.freqs,
            "f1s": self._f1s,
            "f2s": self._f2s,
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
                message="Processing combinations...",
                n_jobs=self._n_jobs,
                verbose=self.verbose,
                prefer="processes",
            ).transpose(1, 0, 2, 3)
        except MemoryError as error:  # pragma: no cover
            raise MemoryError(
                "Memory allocation for the bispectrum computation failed. Try reducing "
                "the sampling frequency of the data, or reduce the precision of the "
                "computation with `pybispectra.set_precision('single')`."
            ) from error

    def _store_results(self) -> None:
        """Store computed bispectrum in an object."""
        self._results = ResultsGeneral(
            self._bispectrum[0],
            self._indices,
            self._f1s,
            self._f2s,
            "Bispectrum",
        )

    @property
    def results(self) -> ResultsGeneral:
        """Bispectrum results."""
        return self._results


class Threenorm(_General):
    """Class for computing the threenorm.

    Parameters
    ----------
    data : ~numpy.ndarray, shape of [epochs, channels, frequencies]
        Fourier coefficients.

    freqs : ~numpy.ndarray, shape of [frequencies]
        Frequencies (in Hz) in :attr:`data`. Frequencies are expected to be evenly
        spaced.

    sampling_freq : int | float
        Sampling frequency (in Hz) of the data from which :attr:`data` was derived.

    verbose : bool (default True)
        Whether or not to report the progress of the processing.

    Methods
    -------
    compute :
        Compute the threenorm, averaged over epochs.

    copy :
        Return a copy of the object.

    Attributes
    ----------
    results : ~pybispectra.utils.ResultsGeneral
        Threenorm results.

    data : ~numpy.ndarray of float, shape of [epochs, channels, frequencies]
        Fourier coefficients.

    freqs : ~numpy.ndarray of float, shape of [frequencies]
        Frequencies (in Hz) in :attr:`data`.

    sampling_freq : int | float
        Sampling frequency (in Hz) of the data from which :attr:`data` was derived.

    verbose : bool
        Whether or not to report the progress of the processing.

    Notes
    -----

    .. versionadded:: 1.2
    """

    def compute(
        self,
        indices: tuple[tuple[int]] | None = None,
        f1s: tuple[int | float] | None = None,
        f2s: tuple[int | float] | None = None,
        n_jobs: int = 1,
    ) -> None:
        r"""Compute the threenorm, averaged over epochs.

        Parameters
        ----------
        indices : tuple of tuple of int, length of 3 | None (default None)
            Indices of the channels :math:`k`, :math:`m`, and :math:`n`, respectively,
            to compute the threenorm for. If :obj:`None`, the threenorm for all channel
            combinations is computed.

        f1s : tuple of int or float, length of 2 | None (default None)
            Start and end lower frequencies to compute the threenorm for, respectively.
            If :obj:`None`, all frequencies are used.

        f2s : tuple of int or float, length of 2 | None (default None)
            Start and end higher frequencies to compute the threenorm for, respectively.
            If :obj:`None`, all frequencies are used.

        n_jobs : int (default ``1``)
            The number of jobs to run in parallel. If ``-1``, all available CPUs are
            used.

        Notes
        -----
        The threenorm, :math:`\textbf{N}`, :footcite:`Shahbazi2014` has the
        general form

        :math:`\textbf{N}_{kmn}(f_1,f_2)=(<|\textbf{k}(f_1)|^3><|\textbf{m} (f_2)|^3>
        <|\textbf{n}(f_2+f_1)|^3>)^{\frac{1}{3}}` ,

        where :math:`kmn` is a combination of signals with Fourier coefficients
        :math:`\textbf{k}`, :math:`\textbf{m}`, and :math:`\textbf{n}`, respectively;
        :math:`f_1` and :math:`f_2` correspond to a lower and higher frequency,
        respectively; and :math:`<>` represents the average value over epochs.

        The threenorm is computed between all values of :attr:`f1s` and :attr:`f2s`. If
        any value of :attr:`f1s` is higher than :attr:`f2s`, a :obj:`numpy.nan` value is
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
            print("Computing threenorm...\n")

        self._compute_threenorm()
        self._store_results()

        if self.verbose:
            print("    ... Threenorm computation finished\n")

    def _reset_attrs(self) -> None:
        """Reset attrs. of the object to prevent interference."""
        super()._reset_attrs()

        self._threenorm = None

    def _compute_threenorm(self) -> None:
        """Compute threenorm between f1s and f2s of seeds and targets."""
        loop_kwargs = [
            {"kmn": np.array([np.array([k, m, n])])}
            for (k, m, n) in zip(self._k, self._m, self._n)
        ]
        static_kwargs = {
            "data": self.data,
            "freqs": self.freqs,
            "f1s": self._f1s,
            "f2s": self._f2s,
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
                message="Processing combinations...",
                n_jobs=self._n_jobs,
                verbose=self.verbose,
                prefer="processes",
            ).transpose(1, 0, 2, 3)
        except MemoryError as error:  # pragma: no cover
            raise MemoryError(
                "Memory allocation for the threenorm computation failed. Try reducing "
                "the sampling frequency of the data, or reduce the precision of the "
                "computation with `pybispectra.set_precision('single')`."
            ) from error

    def _store_results(self) -> None:
        """Store computed threenorm in an object."""
        self._results = ResultsGeneral(
            self._threenorm[0],
            self._indices,
            self._f1s,
            self._f2s,
            "Threenorm",
        )

    @property
    def results(self) -> ResultsGeneral:
        """Threenorm results."""
        return self._results
