"""Tools for handling PAC analysis."""

import numpy as np

from pybispectra.utils import ResultsCFC
from pybispectra.utils._defaults import _precision
from pybispectra.utils._process import (
    _compute_bispectrum,
    _compute_threenorm,
    _ProcessBispectrum,
)
from pybispectra.utils._utils import _compute_in_parallel

np.seterr(divide="ignore", invalid="ignore")  # no warning for NaN division


class PAC(_ProcessBispectrum):
    """Class for computing phase-amplitude coupling (PAC) using the bispectrum.

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

        .. versionadded:: 1.3

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
    results : ~pybispectra.utils.ResultsCFC | tuple of ~pybispectra.utils.ResultsCFC
        PAC results for each of the computed metrics.

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

    _return_nosym = False
    _return_antisym = False
    _return_nonorm = False
    _return_threenorm = False

    _pac_nosym_nonorm = None
    _pac_nosym_threenorm = None
    _pac_antisym_nonorm = None
    _pac_antisym_threenorm = None

    def compute(
        self,
        indices: tuple[tuple[int]] | None = None,
        f1s: tuple[int | float] | None = None,
        f2s: tuple[int | float] | None = None,
        times: tuple[int | float] | None = None,
        antisym: bool | tuple[bool] = False,
        norm: bool | tuple[bool] = False,
        n_jobs: int = 1,
    ) -> None:
        r"""Compute PAC, averaged over epochs.

        Parameters
        ----------
        indices : tuple of tuple of int, length of 2 | None (default None)
            Indices of the seed and target channels, respectively, to compute PAC
            between. If :obj:`None`, coupling between all channels is computed.

        f1s : tuple of int or float, length of 2 | None (default None)
            Start and end lower frequencies to compute PAC on, respectively. If
            :obj:`None`, all frequencies are used.

        f2s : tuple of int or float, length of 2 | None (default None)
            Start and end higher frequencies to compute PAC on, respectively. If
            :obj:`None`, all frequencies are used.

        times : tuple of int or float, length of 2 | None (default None)
            Start and end times (in seconds) to compute PAC on, respectively. If
            :obj:`None`, all timepoints are used.

            .. versionadded:: 1.3

        antisym : bool | tuple of bool (default False)
            Whether to antisymmetrise the PAC results. If a tuple of bool, both forms of
            PAC are computed in turn.

        norm : bool | tuple of bool (default False)
            Whether to normalise the PAC results using the threenorm. If a tuple of
            bool, both forms of PAC are computed in turn.

        n_jobs : int (default ``1``)
            The number of jobs to run in parallel. If ``-1``, all available CPUs are
            used.

        Notes
        -----
        PAC can be computed as the bispectrum, :math:`\textbf{B}`, of signals
        :math:`\textbf{x}` and :math:`\textbf{y}` of the seeds and targets,
        respectively, which has the general form

        :math:`\textbf{B}_{kmn}(f_1,f_2)=<\textbf{k}(f_1)\textbf{m}(f_2)
        \textbf{n}^*(f_2+f_1)>` ,

        where :math:`kmn` is a combination of signals with Fourier coefficients
        :math:`\textbf{k}`, :math:`\textbf{m}`, and :math:`\textbf{n}`, respectively;
        :math:`f_1` and :math:`f_2` correspond to a lower and higher frequency,
        respectively; and :math:`<>` represents the average value over epochs. The
        computation of PAC follows from this :footcite:`Kovach2018`

        :math:`\textbf{B}_{xyy}(f_1,f_2)=<\textbf{x}(f_1)\textbf{y}(f_2)
        \textbf{y}^*(f_2+f_1)>` ,

        :math:`\textrm{PAC}(\textbf{x}_{f_1},\textbf{y}_{f_2})=|\textbf{B}_{xyy}
        (f_1,f_2)|` .

        The bispectrum can be normalised to the bicoherence,
        :math:`\boldsymbol{\mathcal{B}}`, using the threenorm, :math:`\textbf{N}`,
        :footcite:`Shahbazi2014`

        :math:`\textbf{N}_{xyy}(f_1,f_2)=(<|\textbf{x}(f_1)|^3><|\textbf{y}(f_2)|^3>
        <|\textbf{y}(f_2+f_1)|^3>)^{\frac{1}{3}}` ,

        :math:`\boldsymbol{\mathcal{B}}_{xyy}(f_1,f_2)=\Large\frac{\textbf{B}_{xyy}
        (f_1,f_2)}{\textbf{N}_{xyy}(f_1,f_2)}` ,

        :math:`\textrm{PAC}_{\textrm{norm}}(\textbf{x}_{f_1},\textbf{y}_{f_2})=
        |\boldsymbol{\mathcal{B}}_{xyy}(f_1,f_2)|` .

        where the resulting values lie in the range :math:`[0, 1]`. Furthermore, PAC can
        be antisymmetrised by subtracting the results from those found using the
        transposed bispectrum, :math:`\textbf{B}_{yxy}`, :footcite:`Chella2014`

        :math:`\textrm{PAC}_{\textrm{antisym}}(\textbf{x}_{f_1},\textbf{y}_{f_2})=
        |\textbf{B}_{xyy}(f_1,f_2)-\textbf{B}_{yxy}(f_1,f_2)|` .

        A modified approach is used for the normalisation of antisymmetrised PAC
        :footcite:`Chella2016`

        :math:`\textrm{PAC}_{\textrm{norm,antisym}}(\textbf{x}_{f_1},\textbf{y}_{f_2})=
        \Large|\frac{\textbf{B}_{xyy}(f_1,f_2)-\textbf{B}_{yxy}(f_1,f_2)}
        {\textbf{N}_{xyy}(f_1,f_2)+\textbf{N}_{yxy}(f_1,f_2)}|` .

        If the seed and target for a given connection is the same channel and
        antisymmetrisation is being performed, :obj:`numpy.nan` values are returned.

        PAC is computed between all values of ``f1s`` and ``f2s``.

        .. warning::
            For values of ``f1s`` higher than ``f2s`` or where ``f2s + f1s`` exceeds the
            Nyquist frequency, a :obj:`numpy.nan` value is returned.

        References
        ----------
        .. footbibliography::
        """  # noqa: E501
        self._reset_attrs()

        self._sort_metrics(antisym, norm)
        self._sort_indices(indices)
        self._sort_freqs(f1s, f2s)
        self._sort_tmin_tmax(times)
        self._sort_parallelisation(n_jobs)

        if self.verbose:
            print("Computing PAC...\n")

        self._compute_bispectrum()
        if self._return_threenorm:
            self._compute_threenorm()
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
        self._threenorm = None

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

        loop_kwargs = [
            {"data": self.data[:, (seed, target)][..., self._time_idcs]}
            for seed, target in zip(self._seeds, self._targets)
        ]
        static_kwargs = {
            "freqs": self.freqs,
            "f1s": self._f1s,
            "f2s": self._f2s,
            "kmn": kmn,
            "precision": _precision.complex,
        }
        try:
            self._bispectrum = _compute_in_parallel(
                func=_compute_bispectrum,
                loop_kwargs=loop_kwargs,
                static_kwargs=static_kwargs,
                output=np.zeros(
                    (
                        self._n_cons,
                        kmn.shape[0],
                        self._f1s.size,
                        self._f2s.size,
                        self._times.size,
                    ),
                    dtype=_precision.complex,
                ),
                message="Processing connections...",
                n_jobs=self._n_jobs,
                verbose=self.verbose,
                prefer="processes",
            ).transpose(1, 0, 2, 3, 4)
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

        if self._return_antisym:
            # kmm, mkm
            kmn = np.array([np.array([0, 1, 1]), np.array([1, 0, 1])])
        else:
            # kmm
            kmn = np.array([np.array([0, 1, 1])])

        loop_kwargs = [
            {"data": self.data[:, (seed, target)][..., self._time_idcs]}
            for seed, target in zip(self._seeds, self._targets)
        ]
        static_kwargs = {
            "freqs": self.freqs,
            "f1s": self._f1s,
            "f2s": self._f2s,
            "kmn": kmn,
            "precision": _precision.real,
        }
        try:
            self._threenorm = _compute_in_parallel(
                func=_compute_threenorm,
                loop_kwargs=loop_kwargs,
                static_kwargs=static_kwargs,
                output=np.zeros(
                    (
                        self._n_cons,
                        kmn.shape[0],
                        self._f1s.size,
                        self._f2s.size,
                        self._times.size,
                    ),
                    dtype=_precision.real,
                ),
                message="Processing connections...",
                n_jobs=self._n_jobs,
                verbose=self.verbose,
                prefer="processes",
            ).transpose(1, 0, 2, 3, 4)
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
                self._pac_nosym_threenorm = np.abs(
                    self._bispectrum[0] / self._threenorm[0]
                )
            if self._return_antisym:
                self._pac_antisym_threenorm = np.abs(
                    (self._bispectrum[0] - self._bispectrum[1])
                    / (self._threenorm[0] + self._threenorm[1])
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
                    data=self._pac_nosym_nonorm,
                    indices=self._indices,
                    f1s=self._f1s,
                    f2s=self._f2s,
                    times=self._times,
                    name="PAC | Bispectrum",
                )
            )

        if self._pac_nosym_threenorm is not None:
            results.append(
                ResultsCFC(
                    data=self._pac_nosym_threenorm,
                    indices=self._indices,
                    f1s=self._f1s,
                    f2s=self._f2s,
                    times=self._times,
                    name="PAC | Bicoherence",
                )
            )

        if self._pac_antisym_nonorm is not None:
            results.append(
                ResultsCFC(
                    data=self._pac_antisym_nonorm,
                    indices=self._indices,
                    f1s=self._f1s,
                    f2s=self._f2s,
                    times=self._times,
                    name="PAC (antisymmetrised) | Bispectrum",
                )
            )

        if self._pac_antisym_threenorm is not None:
            results.append(
                ResultsCFC(
                    data=self._pac_antisym_threenorm,
                    indices=self._indices,
                    f1s=self._f1s,
                    f2s=self._f2s,
                    times=self._times,
                    name="PAC (antisymmetrised) | Bicoherence",
                )
            )

        self._results = tuple(results)

    @property
    def results(self) -> ResultsCFC | tuple[ResultsCFC]:
        if len(self._results) == 1:
            return self._results[0]
        return self._results
