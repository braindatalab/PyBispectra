"""Tools for handling PAC analysis."""

from copy import deepcopy

import numpy as np
from pqdm.processes import pqdm

from pybispectra.utils import ResultsCFC
from pybispectra.utils._defaults import _precision
from pybispectra.utils._process import (
    _ProcessBispectrum,
    _compute_bispectrum,
    _compute_threenorm,
)


np.seterr(divide="ignore", invalid="ignore")  # no warning for NaN division


class PAC(_ProcessBispectrum):
    """Class for computing phase-amplitude coupling (PAC) using the bispectrum.

    Parameters
    ----------
    data : ~numpy.ndarray, shape of [epochs, channels, frequencies]
        Fourier coefficients.

    freqs : ~numpy.ndarray, shape of [frequencies]
        Frequencies (in Hz) in :attr:`data`.

    sampling_freq : int | float
        Sampling frequency (in Hz) of the data from which :attr:`data` was
        derived.

    verbose : bool (default True)
        Whether or not to report the progress of the processing.

    Methods
    -------
    compute :
        Compute PAC, averaged over epochs.

    copy :
        Return a copy of the object.

    Attributes
    ----------
    results : tuple of ~pybispectra.utils.ResultsCFC
        PAC results for each of the computed metrics.

    data : ~numpy.ndarray of float, shape of [epochs, channels, frequencies]
        Fourier coefficients.

    freqs : ~numpy.ndarray of float, shape of [frequencies]
        Frequencies (in Hz) in :attr:`data`.

    sampling_freq : int | float
        Sampling frequency (in Hz) of the data from which :attr:`data` was
        derived.

    verbose : bool
        Whether or not to report the progress of the processing.
    """

    _return_nosym = False
    _return_antisym = False
    _return_nonorm = False
    _return_threenorm = False

    _bispectrum = None
    _bicoherence = None

    _pac_nosym_nonorm = None
    _pac_nosym_threenorm = None
    _pac_antisym_nonorm = None
    _pac_antisym_threenorm = None

    def compute(
        self,
        indices: tuple[tuple[int]] | None = None,
        f1s: tuple[int | float] | None = None,
        f2s: tuple[int | float] | None = None,
        antisym: bool | tuple[bool] = False,
        norm: bool | tuple[bool] = False,
        n_jobs: int = 1,
    ) -> None:
        r"""Compute PAC, averaged over epochs.

        Parameters
        ----------
        indices : tuple of tuple of int, length of 2 | None (default None)
            Indices of the seed and target channels, respectively, to compute
            PAC between. If :obj:`None`, coupling between all channels is
            computed.

        f1s : tuple of int or float, length of 2 | None (default None)
            Start and end lower frequencies to compute PAC on, respectively. If
            :obj:`None`, all frequencies are used.

        f2s : tuple of int or float, length of 2 | None (default None)
            Start and end higher frequencies to compute PAC on, respectively.
            If :obj:`None`, all frequencies are used.

        antisym : bool | tuple of bool (default False)
            Whether to antisymmetrise the PAC results. If a tuple of bool, both
            forms of PAC are computed in turn.

        norm : bool | tuple of bool (default False)
            Whether to normalise the PAC results using the threenorm. If a
            tuple of bool, both forms of PAC are computed in turn.

        n_jobs : int (default ``1``)
            The number of jobs to run in parallel. If ``-1``, all available
            CPUs are used.

        Notes
        -----
        PAC can be computed as the bispectrum, :math:`\textbf{B}`, of signals
        :math:`\textbf{x}` and :math:`\textbf{y}` of the seeds and targets,
        respectively, which has the general form

        :math:`\textbf{B}_{kmn}(f_1,f_2)=<\textbf{k}(f_1)\textbf{m}(f_2)
        \textbf{n}^*(f_2+f_1)>` ,

        where :math:`kmn` is a combination of signals with Fourier coefficients
        :math:`\textbf{k}`, :math:`\textbf{m}`, and :math:`\textbf{n}`,
        respectively; :math:`f_1` and :math:`f_2` correspond to a lower and
        higher frequency, respectively; and :math:`<>` represents the average
        value over epochs. The computation of PAC follows from this
        :footcite:`Kovach2018`

        :math:`\textbf{B}_{xyy}(f_1,f_2)=<\textbf{x}(f_1)\textbf{y}(f_2)
        \textbf{y}^*(f_2+f_1)>` ,

        :math:`\textrm{PAC}(\textbf{x}_{f_1},\textbf{y}_{f_2})=|
        \textbf{B}_{xyy}(f_1,f_2)|` .

        The bispectrum can be normalised to the bicoherence,
        :math:`\boldsymbol{\mathcal{B}}`, using the threenorm,
        :math:`\textbf{N}`, :footcite:`Shahbazi2014`

        :math:`\textbf{N}_{xyy}(f_1,f_2)=(<|\textbf{x}(f_1)|^3><|\textbf{y}
        (f_2)|^3><|\textbf{y}(f_2+f_1)|^3>)^{\frac{1}{3}}` ,

        :math:`\boldsymbol{\mathcal{B}}_{xyy}(f_1,f_2)=\Large\frac{
        \textbf{B}_{xyy}(f_1,f_2)}{\textbf{N}_{xyy}(f_1,f_2)}` ,

        :math:`\textrm{PAC}_{\textrm{norm}}(\textbf{x}_{f_1},\textbf{y}_{f_2})=
        |\boldsymbol{\mathcal{B}}_{xyy}(f_1,f_2)|` .

        where the resulting values lie in the range :math:`[0, 1]`.
        Furthermore, PAC can be antisymmetrised by subtracting the results from
        those found using the transposed bispectrum, :math:`\textbf{B}_{xyx}`,
        :footcite:`Chella2014`

        :math:`\textrm{PAC}_{\textrm{antisym}}(\textbf{x}_{f_1},\textbf{y}_{
        f_2})=|\textbf{B}_{xyy}-\textbf{B}_{xyx}|` .

        If the seed and target for a given connection is the same channel and
        antisymmetrisation is being performed, :obj:`numpy.nan` values are
        returned.

        PAC is computed between all values of :attr:`f1s` and :attr:`f2s`. If
        any value of :attr:`f1s` is higher than :attr:`f2s`, a :obj:`numpy.nan`
        value is returned.

        References
        ----------
        .. footbibliography::
        """
        self._reset_attrs()

        self._sort_metrics(antisym, norm)
        self._sort_indices(indices)
        self._sort_freqs(f1s, f2s)
        self._sort_parallelisation(n_jobs)

        if self.verbose:
            print("Computing PAC...\n")

        self._compute_bispectrum()
        if self._return_threenorm:
            self._compute_bicoherence()
        self._compute_pac()
        self._store_results()

        if self.verbose:
            print("    ... PAC computation finished\n")

    def _reset_attrs(self) -> None:
        """Reset attrs. of the object to prevent interference."""
        super()._reset_attrs()

        self._return_nosym = False
        self._return_antisym = False
        self._return_nonorm = False
        self._return_threenorm = False

        self._bispectrum = None
        self._bicoherence = None

        self._pac_nosym_nonorm = None
        self._pac_nosym_threenorm = None
        self._pac_antisym_nonorm = None
        self._pac_antisym_threenorm = None

    def _sort_metrics(
        self, antisym: bool | tuple[bool], norm: bool | tuple[bool]
    ) -> None:
        """Sort inputs for the form of results being requested."""
        if not isinstance(antisym, (bool, tuple)):
            raise TypeError("`antisym` must be a bool or tuple of bools.")
        if not isinstance(norm, (bool, tuple)):
            raise TypeError("`norm` must be a bool or tuple of bools.")

        if isinstance(antisym, bool):
            antisym = (antisym,)
        if isinstance(norm, bool):
            norm = (norm,)

        if any(not isinstance(entry, bool) for entry in antisym):
            raise TypeError("Entries of `antisym` must be bools.")
        if any(not isinstance(entry, bool) for entry in norm):
            raise TypeError("Entries of `norm` must be bools.")

        if False in antisym:
            self._return_nosym = True
        if True in antisym:
            self._return_antisym = True

        if False in norm:
            self._return_nonorm = True
        if True in norm:
            self._return_threenorm = True

    def _compute_bispectrum(self) -> None:
        """Compute bispectrum between f1s and f2s of seeds and targets."""
        if self.verbose:
            print("    Computing bispectrum...")

        if self._return_antisym:
            # kmm, mkm
            kmn = np.array([np.array([0, 1, 1]), np.array([1, 0, 1])])
        else:
            # kmm
            kmn = np.array([np.array([0, 1, 1])])

        args = [
            {
                "data": self.data[:, (seed, target)],
                "freqs": self.freqs,
                "f1s": self._f1s,
                "f2s": self._f2s,
                "kmn": kmn,
                "precision": _precision.complex,
            }
            for seed, target in zip(self._seeds, self._targets)
        ]

        # have to average complex value outside of Numba-compiled function
        self._bispectrum = (
            np.array(
                pqdm(
                    args,
                    _compute_bispectrum,
                    self._n_jobs,
                    argument_type="kwargs",
                    desc="Processing connections...",
                    disable=not self.verbose,
                ),
                dtype=_precision.complex,
            )
            .mean(axis=2)
            .transpose(1, 0, 2, 3)
        )

        if self.verbose:
            print("        ... Bispectrum computation finished\n")

    def _compute_bicoherence(self) -> None:
        """Compute bicoherence from the bispectrum using the threenorm."""
        if self.verbose:
            print("    Computing bicoherence...")

        if self._return_antisym:
            # kmm, mkm
            kmn = np.array([np.array([0, 1, 1]), np.array([1, 0, 1])])
        else:
            # kmm
            kmn = np.array([np.array([0, 1, 1])])

        args = [
            {
                "data": self.data[:, (seed, target)],
                "freqs": self.freqs,
                "f1s": self._f1s,
                "f2s": self._f2s,
                "kmn": kmn,
                "precision": _precision.real,
            }
            for seed, target in zip(self._seeds, self._targets)
        ]

        threenorm = np.array(
            pqdm(
                args,
                _compute_threenorm,
                self._n_jobs,
                argument_type="kwargs",
                desc="Processing connections...",
                disable=not self.verbose,
            ),
            dtype=_precision.real,
        ).transpose(1, 0, 2, 3)

        self._bicoherence = self._bispectrum / threenorm

        if self.verbose:
            print("        ... Bicoherence computation finished\n")

    def _compute_pac(self) -> None:
        """Compute PAC results from bispectrum/bicoherence."""
        if self._return_nonorm:
            if self._return_nosym:
                self._pac_nosym_nonorm = np.abs(self._bispectrum[0])
            if self._return_antisym:
                self._pac_antisym_nonorm = np.abs(
                    self._bispectrum[0] - self._bispectrum[1]
                )
                for con_i, seed, target in zip(
                    range(self._n_cons), self._seeds, self._targets
                ):
                    if seed == target:
                        self._pac_antisym_nonorm[con_i] = np.full_like(
                            self._pac_antisym_nonorm[con_i],
                            fill_value=np.nan,
                            dtype=_precision.real,
                        )

        if self._return_threenorm:
            if self._return_nosym:
                self._pac_nosym_threenorm = np.abs(self._bicoherence[0])
            if self._return_antisym:
                self._pac_antisym_threenorm = np.abs(
                    self._bicoherence[0] - self._bicoherence[1]
                )
                for con_i, seed, target in zip(
                    range(self._n_cons), self._seeds, self._targets
                ):
                    if seed == target:
                        self._pac_antisym_threenorm[con_i] = np.full_like(
                            self._pac_antisym_threenorm[con_i],
                            fill_value=np.nan,
                            dtype=_precision.real,
                        )

    def _store_results(self) -> None:
        """Store computed results in objects."""
        results = []

        if self._pac_nosym_nonorm is not None:
            results.append(
                ResultsCFC(
                    self._pac_nosym_nonorm,
                    self._indices,
                    self._f1s,
                    self._f2s,
                    "PAC | Bispectrum",
                )
            )

        if self._pac_nosym_threenorm is not None:
            results.append(
                ResultsCFC(
                    self._pac_nosym_threenorm,
                    self._indices,
                    self._f1s,
                    self._f2s,
                    "PAC | Bicoherence",
                )
            )

        if self._pac_antisym_nonorm is not None:
            results.append(
                ResultsCFC(
                    self._pac_antisym_nonorm,
                    self._indices,
                    self._f1s,
                    self._f2s,
                    "PAC (antisymmetrised) | Bispectrum",
                )
            )

        if self._pac_antisym_threenorm is not None:
            results.append(
                ResultsCFC(
                    self._pac_antisym_threenorm,
                    self._indices,
                    self._f1s,
                    self._f2s,
                    "PAC (antisymmetrised) | Bicoherence",
                )
            )

        self._results = tuple(results)

    @property
    def results(self) -> ResultsCFC | tuple[ResultsCFC]:
        """Return the results.

        Returns
        -------
        results : ~pybispectra.utils.ResultsCFC | tuple of ~pybispectra.utils.ResultsCFC
            The results of the PAC computation returned as a single results
            object (if only one PAC variant was computed) or a tuple of results
            objects.
        """  # noqa: E501
        if len(self._results) == 1:
            return deepcopy(self._results[0])
        return deepcopy(self._results)
