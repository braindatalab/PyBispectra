"""Tools for handling TDE analysis."""

from copy import deepcopy
from typing import Callable

import numpy as np
from numba import njit
from scipy.linalg import hankel

from pybispectra.utils import ResultsTDE
from pybispectra.utils._defaults import _precision
from pybispectra.utils._process import _ProcessBispectrum
from pybispectra.utils._utils import _compute_in_parallel, _number_like, _int_like


class TDE(_ProcessBispectrum):
    """Class for computing time delay estimation (TDE) using the bispectrum.

    Parameters
    ----------
    data : ~numpy.ndarray, shape of [epochs, channels, frequencies]
        Fourier coefficients. Must contain a coefficient for the zero frequency.
        Coefficients should be computed with the number of points equal to twice the
        number of timepoints in each epoch of the original data plus one (i.e.
        ``n_points=2 * n_times + 1`` in :func:`pybispectra.utils.compute_fft`).

    freqs : ~numpy.ndarray, shape of [frequencies]
        Frequencies (in Hz) in ``data``. Frequencies are expected to be evenly spaced.

    sampling_freq : int | float
        Sampling frequency (in Hz) of the data from which ``data`` was derived.

    verbose : bool (default True)
        Whether or not to report the progress of the processing.

    Methods
    -------
    compute :
        Compute TDE, averaged over epochs.

    copy :
        Return a copy of the object.

    Attributes
    ----------
    results : ~pybispectra.utils.ResultsTDE | tuple of ~pybispectra.utils.ResultsTDE
        TDE results for each of the computed metrics.

    data : ~numpy.ndarray, shape of [epochs, channels, frequencies]
        Fourier coefficients, with negative frequencies appended.

    freqs : ~numpy.ndarray, shape of [frequencies]
        Frequencies (in Hz) in ``data``, with negative frequencies appended.

    sampling_freq : int | float
        Sampling frequency (in Hz) of the data from which ``data`` was derived.

    verbose : bool
        Whether or not to report the progress of the processing.

    Notes
    -----
    TDE with the bispectrum requires the Fourier coefficients of the negative
    frequencies of the original signals, however since these are expected to be
    real-valued, they can be inferred from the positive frequencies. Accordingly, only
    the coefficients corresponding to the zero and positive frequencies should be passed
    to ``data``.

    It is recommended to compute the Fourier coefficients with ``n_points=2 * n_times +
    1``. Using a smaller number of points than this will reduce the time range in which
    a delay estimate can be generated below that of the length of the epochs.
    Furthermore, a larger number of points than this will only artificially increase the
    time range in which a delay estimate can be generated beyond the length of the
    epochs.
    """

    _data_ndims: tuple = (3,)  # [epochs, channels, frequencies]

    _freq_masks: np.ndarray = None
    _freq_bands: tuple[tuple[float]] = None

    _return_nosym: bool = False
    _return_antisym: bool = False
    _return_method_i: bool = False
    _return_method_ii: bool = False
    _return_method_iii: bool = False
    _return_method_iv: bool = False

    _bispectrum: np.ndarray = None

    _tde_i_nosym: np.ndarray = None
    _tde_i_antisym: np.ndarray = None
    _tde_ii_nosym: np.ndarray = None
    _tde_ii_antisym: np.ndarray = None
    _tde_iii_nosym: np.ndarray = None
    _tde_iii_antisym: np.ndarray = None
    _tde_iv_nosym: np.ndarray = None
    _tde_iv_antisym: np.ndarray = None

    _kmn: dict = {
        "xxx": (0, 0, 0),
        "yyy": (1, 1, 1),
        "xyx": (0, 1, 0),
        "yxx": (1, 0, 0),
    }
    _xyz: dict = None

    def __init__(
        self,
        data: np.ndarray,
        freqs: np.ndarray,
        sampling_freq: int | float,
        verbose: bool = True,
    ) -> None:  # noqa: D107
        super().__init__(data, freqs, sampling_freq, times=None, verbose=verbose)
        self._sort_fft_coeffs()

    def _sort_fft_coeffs(self) -> None:
        """Check the freqs. are appropriate and add the negative freqs."""
        if self.freqs[0] != 0.0:
            raise ValueError("The first entry of `freqs` must be 0.")

        self._data = np.concatenate(
            (self._data, np.conjugate(self._data[..., 1:][..., ::-1])), axis=2
        )
        self.freqs = np.concatenate((self.freqs, -self.freqs[1:][::-1]), axis=0)

        self._n_unique_freqs = np.unique(np.abs(self.freqs)).size

    def compute(
        self,
        indices: tuple[tuple[int]] | None = None,
        fmin: int | float | tuple[int | float] = 0.0,
        fmax: int | float | tuple[int | float] = np.inf,
        antisym: bool | tuple[bool] = False,
        method: int | tuple[int] = 1,
        n_jobs: int = 1,
    ) -> None:
        r"""Compute TDE, averaged over epochs.

        Parameters
        ----------
        indices : tuple of tuple of int, length of 2 | None (default None)
            Indices of the seed and target channels, respectively, to compute TDE
            between. If :obj:`None`, coupling between all channels is computed.

        fmin : int | float | tuple of int or float (default ``0.0``)
            The low frequency of interest (in Hz) to compute time delays for. If a tuple
            of float, specifies the low frequencies for each frequency band of interest
            (must have the same length as ``fmax``).

        fmax : int | float | tuple of int or float (default numpy.inf)
            The high frequency of interest (in Hz) to compute time delays for. If a
            tuple of float, specifies the high frequencies for each frequency band of
            interest (must have the same length as ``fmin``).

        antisym : bool | tuple of bool (default False)
            Whether to antisymmetrise the PAC results. If a tuple of bool, both forms of
            PAC are computed in turn.

        method : int | tuple of int (default ``1``)
            The method to use to compute TDE :footcite:`Nikias1988`. Can include ``[1,
            2, 3, 4]``.

        n_jobs : int (default ``1``)
            The number of jobs to run in parallel. If ``-1``, all available CPUs are
            used.

        Notes
        -----
        TDE can be computed from the bispectrum, :math:`\textbf{B}`, of signals
        :math:`\textbf{x}` and :math:`\textbf{y}` of the seeds and targets,
        respectively, which has the general form

        :math:`\textbf{B}_{kmn}(f_1,f_2)=<\textbf{k}(f_1)\textbf{m}(f_2)
        \textbf{n}^*(f_2+f_1)>` ,

        where :math:`kmn` is a combination of signals with Fourier coefficients
        :math:`\textbf{k}`, :math:`\textbf{m}`, and :math:`\textbf{n}`, respectively;
        :math:`f_1` and :math:`f_2` correspond to a lower and higher frequency,
        respectively; and :math:`<>` represents the average value over epochs. When
        computing TDE, information from :math:`\textbf{n}` is taken not only from the
        positive frequencies, but also the negative frequencies.

        Four methods exist for computing TDE based on the bispectrum
        :footcite:`Nikias1988`. The fundamental equation is as follows

        :math:`\textrm{TDE}_{xy}(\tau)=\int_{-\pi}^{+\pi}\int_{-\pi}^{+\pi}\textbf{I}
        (\textbf{x}_{f_1},\textbf{y}_{f_2})e^{-if_1\tau}df_1df_2` ,

        where :math:`\textbf{I}` varies depending on the method; and :math:`\tau` is a
        given time delay. Phase information of the signals is extracted from the
        bispectrum in two variants used by the different methods:

        :math:`\boldsymbol{\phi}(\textbf{x}_{f_1},\textbf{y}_{f_2})=\boldsymbol{\varphi}
        _{\textbf{B}_{xyx}} (f_1,f_2)-\boldsymbol{\varphi}_{\textbf{B}_{xxx}}
        (f_1,f_2)` ;

        :math:`\boldsymbol{\phi}'(\textbf{x}_{f_1},\textbf{y}_{f_2})=
        \boldsymbol{\varphi}_{\textbf{B}_{xyx}}(f_1,f_2)-\frac{1}{2}(
        \boldsymbol{\varphi}_{\textbf{B}_{xxx}}(f_1,f_2) + \boldsymbol{\varphi}_{
        \textbf{B}_{yyy}}(f_1,f_2))` .

        **Method I**:
        :math:`\textbf{I}(\textbf{x}_{f_1},\textbf{y}_{f_2})=e^{i\boldsymbol{\phi}
        (\textbf{x}_{f_1},\textbf{y}_{f_2})}`

        **Method II**:
        :math:`\textbf{I}(\textbf{x}_{f_1},\textbf{y}_{f_2})=e^{i\boldsymbol{\phi}'
        (\textbf{x}_{f_1},\textbf{y}_{f_2})}`

        **Method III**:
        :math:`\textbf{I}(\textbf{x}_{f_1},\textbf{y}_{f_2})=\Large \frac{\textbf{B}_
        {xyx}(f_1,f_2)}{\textbf{B}_{xxx}(f_1,f_2)}`

        **Method IV**:
        :math:`\textbf{I}(\textbf{x}_{f_1},\textbf{y}_{f_2})=\Large \frac{|\textbf{B}_
        {xyx}(f_1,f_2)|e^{i\boldsymbol{\phi}'(\textbf{x}_{f_1},\textbf{y}_{f_2})}}{
        \sqrt{|\textbf{B}_{xxx}(f_1,f_2)||\textbf{B}_{yyy}(f_1,f_2)|}}`

        where :math:`\boldsymbol{\varphi}_{\textbf{B}}` is the phase of the bispectrum.
        All four methods aim to capture the phase difference between :math:`\textbf{x}`
        and :math:`\textbf{y}`. Method I involves the extraction of phase spectrum
        periodicity and monotony, with method III involving an additional amplitude
        weighting from the bispectrum of :math:`\textbf{x}`. Method II instead relies on
        a combination of phase spectra of the different frequency components, with
        method IV containing an additional amplitude weighting from the bispectrum of
        :math:`\textbf{x}` and :math:`\textbf{y}`. No single method is superior to
        another.

        Antisymmetrisation of the bispectrum is implemented as the replacement of
        :math:`\textbf{B}_{xyx}` with :math:`(\textbf{B}_{xyx} - \textbf{B}_{yxx})` in
        the above equations :footcite:`JurharPrePrint`.

        If the seed and target for a given connection is the same channel, an error is
        raised.

        References
        ----------
        .. footbibliography::
        """
        super()._reset_attrs()

        self._sort_freq_bands(fmin, fmax)
        self._sort_metrics(antisym, method)
        self._sort_indices(indices)
        self._sort_parallelisation(n_jobs)

        if self.verbose:
            print("Computing TDE...\n")

        self._compute_bispectrum()
        self._compute_tde()
        self._store_results()

        self._reset_attrs()

        if self.verbose:
            print("    ... TDE computation finished\n")

    def _reset_attrs(self) -> None:
        """Reset attrs. of the object to prevent interference."""
        self._freq_bands = None
        self._freq_masks = None

        self._return_nosym = False
        self._return_antisym = False
        self._return_method_i = False
        self._return_method_ii = False
        self._return_method_iii = False
        self._return_method_iv = False

        self._bispectrum = None

        self._tde_i_nosym = None
        self._tde_i_antisym = None
        self._tde_ii_nosym = None
        self._tde_ii_antisym = None
        self._tde_iii_nosym = None
        self._tde_iii_antisym = None
        self._tde_iv_nosym = None
        self._tde_iv_antisym = None

        self._xyz = None

    def _sort_freq_bands(
        self,
        fmin: int | float | tuple[int | float],
        fmax: int | float | tuple[int | float],
    ) -> None:
        """Sort inputs for the frequency bounds."""
        if not isinstance(fmin, _number_like + (tuple,)):
            raise TypeError("`fmin` must be an int, float, or tuple.")
        if not isinstance(fmax, _number_like + (tuple,)):
            raise TypeError("`fmax` must be an int, float, or tuple.")

        if isinstance(fmin, _number_like):
            fmin = (fmin,)
        if isinstance(fmax, _number_like):
            fmax = (fmax,)

        new_fmax = []
        for this_fmax in fmax:
            if this_fmax == np.inf:
                this_fmax = self.freqs.max()
            new_fmax.append(this_fmax)
        fmax = tuple(new_fmax)

        if len(fmin) != len(fmax):
            raise ValueError("`fmin` and `fmax` must have the same length.")
        if any(freq < 0 for freq in fmin):
            raise ValueError("Entries of `fmin` must be >= 0.")
        if any(freq > self.sampling_freq / 2 for freq in fmax):
            raise ValueError("Entries of `fmax` must be <= the Nyquist frequency.")
        if any(this_fmin > this_fmax for this_fmin, this_fmax in zip(fmin, fmax)):
            raise ValueError(
                "At least one entry of `fmin` is > the corresponding entry of `fmax`."
            )

        freq_masks = []
        freq_bands = []
        for this_fmin, this_fmax in zip(fmin, fmax):
            freq_mask = np.zeros((self._n_unique_freqs,), dtype=np.int32)
            freq_mask[
                np.nonzero((self.freqs >= this_fmin) & (self.freqs <= this_fmax))
            ] = 1
            if np.all(freq_mask == 0):
                raise ValueError(
                    "No frequencies are present in the data for the range "
                    f"({this_fmin}, {this_fmax})."
                )
            freq_masks.append(freq_mask)
            freq_bands.append((this_fmin, this_fmax))

        self._freq_masks = np.array(freq_masks)
        self._freq_bands = tuple(freq_bands)

    def _sort_metrics(
        self, antisym: bool | tuple[bool], method: int | tuple[int]
    ) -> None:
        """Sort inputs for the form of results being requested."""
        if not isinstance(antisym, (bool, tuple)):
            raise TypeError("`antisym` must be a bool or tuple of bools.")
        if not isinstance(method, _int_like + (tuple,)):
            raise TypeError("`method` must be an int or tuple of ints.")

        if isinstance(antisym, bool):
            antisym = (antisym,)
        if isinstance(method, _int_like):
            method = (method,)

        if any(not isinstance(entry, bool) for entry in antisym):
            raise TypeError("Entries of `antisym` must be bools.")
        supported_meth = [1, 2, 3, 4]
        if any(entry not in supported_meth for entry in method):
            raise ValueError("The value of `method` is not recognised.")

        if False in antisym:
            self._return_nosym = True
        if True in antisym:
            self._return_antisym = True

        if 1 in method:
            self._return_method_i = True
        if 2 in method:
            self._return_method_ii = True
        if 3 in method:
            self._return_method_iii = True
        if 4 in method:
            self._return_method_iv = True

    def _sort_indices(self, indices: tuple[tuple[int]] | None) -> None:
        """Sort seed-target indices inputs."""
        if indices is None:
            indices = tuple(
                map(tuple, np.array(np.triu_indices(self._n_chans, 1)).tolist())
            )
        if not isinstance(indices, tuple):
            raise TypeError("`indices` must be a tuple.")
        if len(indices) != 2:
            raise ValueError("`indices` must have length of 2.")
        self._indices = indices

        seeds = indices[0]
        targets = indices[1]
        for group_idcs in (seeds, targets):
            if not isinstance(group_idcs, tuple):
                raise TypeError("Entries of `indices` must be tuples.")
            if any(not isinstance(idx, _int_like) for idx in group_idcs):
                raise TypeError(
                    "Entries for seeds and targets in `indices` must be ints."
                )
            if any(idx < 0 or idx >= self._n_chans for idx in group_idcs):
                raise ValueError(
                    "`indices` contains indices for channels not present in the data."
                )
        if len(seeds) != len(targets):
            raise ValueError("Entries of `indices` must have equal length.")
        if any(seed == target for seed, target in zip(indices[0], indices[1])):
            raise ValueError(
                "Seeds and targets in `indices` must not be the same channel for any "
                "connection."
            )
        self._seeds = seeds
        self._targets = targets

        self._n_cons = len(seeds)

    def _compute_bispectrum(self) -> None:
        """Compute bispectrum between f1s and f2s of seeds and targets."""
        if self.verbose:
            print("    Computing bispectrum...")

        self._xyz = deepcopy(self._kmn)
        if not self._return_method_ii and not self._return_method_iv:
            del self._xyz["yyy"]
        if not self._return_antisym:
            del self._xyz["yxx"]
        kmn = np.array(list(self._xyz.values()))

        hankel_freq_mask = hankel(
            np.arange(self._n_unique_freqs),
            np.arange(self._n_unique_freqs - 1, 2 * self._n_unique_freqs - 1),
        )

        loop_kwargs = [
            {"data": self._data[:, (seed, target)]}
            for seed, target in zip(self._seeds, self._targets)
        ]
        static_kwargs = {
            "hankel_freq_mask": hankel_freq_mask,
            "kmn": kmn,
            "precision": _precision.complex,
        }
        try:
            self._bispectrum = _compute_in_parallel(
                func=_compute_bispectrum_tde,
                loop_kwargs=loop_kwargs,
                static_kwargs=static_kwargs,
                output=np.zeros(
                    (self._n_cons, kmn.shape[0], *hankel_freq_mask.shape),
                    dtype=_precision.complex,
                ),
                message="Processing connections...",
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

        if self.verbose:
            print("        ... Bispectrum computation finished\n")

    def _compute_tde(self) -> None:
        """Compute TDE results from bispectra."""
        if self.verbose:
            print("    Computing TDE...")

        self._compute_times()

        if self._return_nosym:
            self._compute_tde_nosym()

        if self._return_antisym:
            self._compute_tde_antisym()

        if self.verbose:
            print("        ... TDE computation finished\n")

    def _compute_tde_nosym(self) -> None:
        """Compute unsymmetrised TDE."""
        B_xxx = self._bispectrum[list(self._xyz.keys()).index("xxx")]

        if self._return_method_ii or self._return_method_iv:
            B_yyy = self._bispectrum[list(self._xyz.keys()).index("yyy")]

        B_xyx = self._bispectrum[list(self._xyz.keys()).index("xyx")]

        if self._return_method_i:
            self._tde_i_nosym = self._compute_tde_form_parallel(
                _compute_tde_i, {"B_xyx": B_xyx, "B_xxx": B_xxx}
            )
        if self._return_method_ii:
            self._tde_ii_nosym = self._compute_tde_form_parallel(
                _compute_tde_ii, {"B_xyx": B_xyx, "B_xxx": B_xxx, "B_yyy": B_yyy}
            )
        if self._return_method_iii:
            self._tde_iii_nosym = self._compute_tde_form_parallel(
                _compute_tde_iii, {"B_xyx": B_xyx, "B_xxx": B_xxx}
            )
        if self._return_method_iv:
            self._tde_iv_nosym = self._compute_tde_form_parallel(
                _compute_tde_iv, {"B_xyx": B_xyx, "B_xxx": B_xxx, "B_yyy": B_yyy}
            )

    def _compute_tde_antisym(self) -> None:
        """Compute antisymmetrised TDE."""
        B_xxx = self._bispectrum[list(self._xyz.keys()).index("xxx")]

        if self._return_method_ii or self._return_method_iv:
            B_yyy = self._bispectrum[list(self._xyz.keys()).index("yyy")]

        B_xyx = (
            self._bispectrum[list(self._xyz.keys()).index("xyx")]
            - self._bispectrum[list(self._xyz.keys()).index("yxx")]
        )

        if self._return_method_i:
            self._tde_i_antisym = self._compute_tde_form_parallel(
                _compute_tde_i, {"B_xyx": B_xyx, "B_xxx": B_xxx}
            )
        if self._return_method_ii:
            self._tde_ii_antisym = self._compute_tde_form_parallel(
                _compute_tde_ii, {"B_xyx": B_xyx, "B_xxx": B_xxx, "B_yyy": B_yyy}
            )
        if self._return_method_iii:
            self._tde_iii_antisym = self._compute_tde_form_parallel(
                _compute_tde_iii, {"B_xyx": B_xyx, "B_xxx": B_xxx}
            )
        if self._return_method_iv:
            self._tde_iv_antisym = self._compute_tde_form_parallel(
                _compute_tde_iv, {"B_xyx": B_xyx, "B_xxx": B_xxx, "B_yyy": B_yyy}
            )

    def _compute_tde_form_parallel(self, func: Callable, kwargs: dict) -> np.ndarray:
        """Compute TDE in parallel across connections for a single form.

        Parameters
        ----------
        func : Callable
            TDE computation function to parallelise.

        kwargs : dict
            Arguments to pass to ``func``.

        Returns
        -------
        tde : numpy.ndarray of float, shape of [connections, times]
            Time delay estimates.
        """
        assert isinstance(kwargs, dict), (
            "PyBispectra Internal Error: `kwargs` for parallelisation must be a dict. "
            "Please contact the PyBispectra developers."
        )

        loop_kwargs = []
        for con_i in range(self._n_cons):
            loop_kwargs.append({key: value[con_i] for key, value in kwargs.items()})
        static_kwargs = {"freq_masks": self._freq_masks}
        return _compute_in_parallel(
            func=func,
            loop_kwargs=loop_kwargs,
            static_kwargs=static_kwargs,
            output=np.zeros(
                (self._n_cons, len(self._freq_masks), len(self._times)),
                dtype=_precision.real,
            ),
            message="Processing connections...",
            n_jobs=self._n_jobs,
            verbose=self.verbose,
            prefer="processes",
        )

    def _compute_times(self) -> None:
        """Compute timepoints (in ms) in the results."""
        epoch_dur = (self._n_unique_freqs - 1) / self.sampling_freq
        self._times = (
            np.linspace(
                -epoch_dur,
                epoch_dur,
                2 * self._n_unique_freqs - 1,
                dtype=_precision.real,
            )
            * 1000
        )

    def _store_results(self) -> None:
        """Store computed results in objects."""
        results = []

        if self._tde_i_nosym is not None:
            results.append(
                ResultsTDE(
                    data=self._tde_i_nosym,
                    indices=self._indices,
                    times=self._times,
                    freq_bands=self._freq_bands,
                    name="TDE | Method I",
                )
            )
        if self._tde_ii_nosym is not None:
            results.append(
                ResultsTDE(
                    data=self._tde_ii_nosym,
                    indices=self._indices,
                    times=self._times,
                    freq_bands=self._freq_bands,
                    name="TDE | Method II",
                )
            )
        if self._tde_iii_nosym is not None:
            results.append(
                ResultsTDE(
                    data=self._tde_iii_nosym,
                    indices=self._indices,
                    times=self._times,
                    freq_bands=self._freq_bands,
                    name="TDE | Method III",
                )
            )
        if self._tde_iv_nosym is not None:
            results.append(
                ResultsTDE(
                    data=self._tde_iv_nosym,
                    indices=self._indices,
                    times=self._times,
                    freq_bands=self._freq_bands,
                    name="TDE | Method IV",
                )
            )

        if self._tde_i_antisym is not None:
            results.append(
                ResultsTDE(
                    data=self._tde_i_antisym,
                    indices=self._indices,
                    times=self._times,
                    freq_bands=self._freq_bands,
                    name="TDE (antisymmetrised) | Method I",
                )
            )
        if self._tde_ii_antisym is not None:
            results.append(
                ResultsTDE(
                    data=self._tde_ii_antisym,
                    indices=self._indices,
                    times=self._times,
                    freq_bands=self._freq_bands,
                    name="TDE (antisymmetrised) | Method II",
                )
            )
        if self._tde_iii_antisym is not None:
            results.append(
                ResultsTDE(
                    data=self._tde_iii_antisym,
                    indices=self._indices,
                    times=self._times,
                    freq_bands=self._freq_bands,
                    name="TDE (antisymmetrised) | Method III",
                )
            )
        if self._tde_iv_antisym is not None:
            results.append(
                ResultsTDE(
                    data=self._tde_iv_antisym,
                    indices=self._indices,
                    times=self._times,
                    freq_bands=self._freq_bands,
                    name="TDE (antisymmetrised) | Method IV",
                )
            )

        self._results = tuple(results)

    @property
    def results(self) -> ResultsTDE | tuple[ResultsTDE]:
        if len(self._results) == 1:
            return self._results[0]
        return self._results


@njit
def _compute_bispectrum_tde(
    data: np.ndarray, hankel_freq_mask: np.ndarray, kmn: np.ndarray, precision: type
) -> np.ndarray:  # pragma: no cover
    """Compute the bispectrum for a single connection for use in TDE.

    Parameters
    ----------
    data : numpy.ndarray, shape of [epochs, 2, frequencies]
        Fourier coefficients, where the second dimension contains the data for the seed
        and target channel of a single connection, respectively. Contains coefficients
        for the zero frequency, the positive frequencies, and the negative frequencies,
        respectively.

    hankel_freq_mask : numpy.ndarray, shape of [frequencies, frequencies]
        Hankel matrix to use as a frequency mask for the frequencies in channel n of
        ``data``, where ``fs`` is the zero and positive frequencies. Can be generated
        with ``scipy.linalg.hankel(c=numpy.arange(0, fs), r=(numpy.arange(fs-1 :
        fs*2))``.

    kmn : numpy.ndarray of int, shape of [x, 3]
        Array of variable length (x) of arrays, where each sub-array contains the k, m,
        and n channel indices in ``data``, respectively, to compute the bispectrum for.

    precision : type
        Precision to use for the computation. Either ``numpy.complex64`` (single) or
        ``numpy.complex128`` (double).

    Returns
    -------
    results : numpy.ndarray, shape of [x, frequencies, frequencies]
        Complex-valued array containing the bispectrum of a single connection, where the
        first dimension corresponds to the different channel indices given in ``kmn``.

    Notes
    -----
    No checks on the input data are performed for speed.
    """
    n_unique_freqs = hankel_freq_mask.shape[0]
    results = np.zeros((kmn.shape[0], n_unique_freqs, n_unique_freqs), dtype=precision)

    for kmn_i, (k, m, n) in enumerate(kmn):
        for epoch_data in data:
            # No arrays as indices in Numba, so loop over to pass int indices
            hankel_n = np.empty_like(hankel_freq_mask, dtype=precision)
            for row_i in range(n_unique_freqs):
                for col_i in range(n_unique_freqs):
                    hankel_n[row_i, col_i] = epoch_data[
                        n, hankel_freq_mask[row_i, col_i]
                    ]

            results[kmn_i] += np.multiply(
                np.transpose(np.expand_dims(epoch_data[k, :n_unique_freqs], 0)),
                np.multiply(epoch_data[m, :n_unique_freqs], np.conjugate(hankel_n)),
            )

    return np.divide(results, data.shape[0]).astype(precision)


def _compute_tde_i(
    B_xyx: np.ndarray, B_xxx: np.ndarray, freq_masks: np.ndarray
) -> np.ndarray:
    """Compute TDE from bispectra with method I for a single connection.

    Parameters
    ----------
    B_xyx : numpy.ndarray, shape of [frequencies, frequencies]
        Bispectrum for channel combination ``xyx``.

    B_xxx : numpy.ndarray, shape of [frequencies, frequencies]
        Bispectrum for channel combination ``xxx``.

    freq_masks : numpy.ndarray, shape of [frequency_bands, frequencies]
        Indices masks for the frequencies to use in each frequency band.

    Returns
    -------
    tde : numpy.ndarray, shape of [frequency_bands, times]
        Time delay estimates for each frequency band.

    Notes
    -----
    No checks on the input data are performed for speed.
    """
    phi = np.angle(B_xyx) - np.angle(B_xxx)
    I = np.exp(1j * phi)

    return _compute_tde_from_I(I, freq_masks)


def _compute_tde_ii(
    B_xyx: np.ndarray,
    B_xxx: np.ndarray,
    B_yyy: np.ndarray,
    freq_masks: np.ndarray,
) -> np.ndarray:
    """Compute TDE from bispectra with method II for a single connection.

    Parameters
    ----------
    B_xyx : numpy.ndarray, shape of [frequencies, frequencies]
        Bispectrum for channel combination ``xyx``.

    B_xxx : numpy.ndarray, shape of [frequencies, frequencies]
        Bispectrum for channel combination ``xxx``.

    B_yyy : numpy.ndarray, shape of [frequencies, frequencies]
        Bispectrum for channel combination ``yyy``.

    freq_masks : numpy.ndarray, shape of [frequency_bands, frequencies]
        Indices masks for the frequencies to use in each frequency band.

    Returns
    -------
    tde : numpy.ndarray, shape of [frequency_bands, times]
        Time delay estimates for each frequency band.

    Notes
    -----
    No checks on the input data are performed for speed.
    """
    phi_prime = np.angle(B_xyx) - 0.5 * (np.angle(B_xxx) + np.angle(B_yyy))
    I = np.exp(1j * phi_prime)

    return _compute_tde_from_I(I, freq_masks)


def _compute_tde_iii(
    B_xyx: np.ndarray, B_xxx: np.ndarray, freq_masks: np.ndarray
) -> np.ndarray:
    """Compute TDE from bispectra with method III for a single connection.

    Parameters
    ----------
    B_xyx : numpy.ndarray, shape of [frequencies, frequencies]
        Bispectrum for channel combination ``xyx``.

    B_xxx : numpy.ndarray, shape of [frequencies, frequencies]
        Bispectrum for channel combination ``xxx``.

    freq_masks : numpy.ndarray, shape of [frequency_bands, frequencies]
        Indices masks for the frequencies to use in each frequency band.

    Returns
    -------
    tde : numpy.ndarray, shape of [frequency_bands, times]
        Time delay estimates for each frequency band.

    Notes
    -----
    No checks on the input data are performed for speed.
    """
    I = np.divide(B_xyx, B_xxx)

    return _compute_tde_from_I(I, freq_masks)


def _compute_tde_iv(
    B_xyx: np.ndarray,
    B_xxx: np.ndarray,
    B_yyy: np.ndarray,
    freq_masks: np.ndarray,
) -> np.ndarray:
    """Compute TDE from bispectra with method IV for a single connection.

    Parameters
    ----------
    B_xyx : numpy.ndarray, shape of [frequencies, frequencies]
        Bispectrum for channel combination ``xyx``.

    B_xxx : numpy.ndarray, shape of [frequencies, frequencies]
        Bispectrum for channel combination ``xxx``.

    B_yyy : numpy.ndarray, shape of [frequencies, frequencies]
        Bispectrum for channel combination ``yyy``.

    freq_masks : numpy.ndarray, shape of [frequency_bands, frequencies]
        Indices masks for the frequencies to use in each frequency band.

    Returns
    -------
    tde : numpy.ndarray, shape of [frequency_bands, times]
        Time delay estimates for each frequency band.

    Notes
    -----
    No checks on the input data are performed for speed.
    """
    phi_prime = np.angle(B_xyx) - 0.5 * (np.angle(B_xxx) + np.angle(B_yyy))
    I = np.divide(
        np.multiply(np.abs(B_xyx), np.exp(1j * phi_prime)),
        np.sqrt(np.multiply(np.abs(B_xxx), np.abs(B_yyy))),
    )

    return _compute_tde_from_I(I, freq_masks)


def _compute_tde_from_I(I: np.ndarray, freq_masks: np.ndarray) -> np.ndarray:
    """Compute TDE from the matrix I for a single connection."""
    tde = np.full(
        (freq_masks.shape[0], I.shape[0] * 2 - 1),
        fill_value=np.nan,
        dtype=_precision.real,
    )
    for fband_i, freq_mask in enumerate(freq_masks):
        I_fband = freq_mask[:, np.newaxis] * (freq_mask * I)
        I_fband = np.nansum(I_fband, axis=0)
        I_fband = np.concatenate(
            (I_fband, np.zeros((I.shape[0] - 1), dtype=_precision.complex))
        )

        tde[fband_i] = np.abs(np.fft.fftshift(np.fft.ifft(I_fband))).astype(
            _precision.real
        )

    return tde
