"""Tools for handling TDE analysis."""

from copy import deepcopy
from typing import Callable

import numpy as np
from pqdm.processes import pqdm
from numba import njit
from scipy.linalg import hankel

from pybispectra.utils import ResultsTDE
from pybispectra.utils._process import _ProcessBispectrum


class TDE(_ProcessBispectrum):
    """Class for computing time delay estimation (TDE) using the bispectrum.

    Parameters
    ----------
    data : ~numpy.ndarray, shape of [epochs, channels, frequencies]
        Fourier coefficients. Must contain coefficients for the zero frequency,
        a set of positive frequencies, and all corresponding negative
        frequencies, like that obtained from
        :func:`~pybispectra.utils.compute_fft` with ``return_neg_freqs=True``.

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
        Compute TDE, averaged over epochs.

    copy :
        Return a copy of the object.

    Attributes
    ----------
    results : tuple of ~pybispectra.utils.ResultsTDE
        TDE results for each of the computed metrics.

    data : ~numpy.ndarray, shape of [epochs, channels, frequencies]
        Fourier coefficients.

    freqs : ~numpy.ndarray, shape of [frequencies]
        Frequencies (in Hz) in :attr:`data`.

    sampling_freq : int | float
        Sampling frequency (in Hz) of the data from which :attr:`data` was
        derived.

    verbose : bool
        Whether or not to report the progress of the processing.
    """

    _allow_neg_freqs = True

    _return_nosym = False
    _return_antisym = False
    _return_method_i = False
    _return_method_ii = False
    _return_method_iii = False
    _return_method_iv = False

    _bispectra = None

    _tde_i_nosym = None
    _tde_i_antisym = None
    _tde_ii_nosym = None
    _tde_ii_antisym = None
    _tde_iii_nosym = None
    _tde_iii_antisym = None
    _tde_iv_nosym = None
    _tde_iv_antisym = None

    _kmn = {
        "xxx": (0, 0, 0),
        "yyy": (1, 1, 1),
        "xyx": (0, 1, 0),
        "xxy": (0, 0, 1),
        "yxx": (1, 0, 0),
    }
    _xyz = None

    _times = None

    def __init__(
        self,
        data: np.ndarray,
        freqs: np.ndarray,
        sampling_freq: int | float,
        verbose: bool = True,
    ) -> None:  # noqa: D107
        super().__init__(data, freqs, sampling_freq, verbose)
        self._sort_freqs_structure()

    def _sort_freqs_structure(self) -> None:
        """Check the freqs. have the appropriate structure."""
        if self.freqs[0] != 0.0:
            raise ValueError("The first entry of `freqs` must be 0.")

        if len(self.freqs) % 2 == 0:  # should be odd (i.e. pos + neg + 0)
            raise ValueError(
                "`freqs` must have an odd number of entries, consisting of "
                "the positive frequencies, the corresponding negative "
                "frequencies, and the zero frequency."
            )

        mid_idx = len(self.freqs) // 2
        if np.any(
            self.freqs[1 : mid_idx + 1] != self.freqs[::-1][:mid_idx] * -1
        ):
            raise ValueError(
                "Each positive frequency in `freqs` must have a corresponding "
                "negative frequency."
            )
        self._n_unique_freqs = mid_idx + 1  # pos. freqs. + zero freq.

    def compute(
        self,
        indices: tuple[tuple[int]] | None = None,
        antisym: bool | tuple[bool] = False,
        method: int | tuple[int] = 1,
        n_jobs: int = 1,
    ) -> None:
        r"""Compute TDE, averaged over epochs.

        Parameters
        ----------
        indices : tuple of tuple of int, length of 2 | None (default None)
            Indices of the seed and target channels, respectively, to compute
            TDE between. If :obj:`None`, coupling between all channels is
            computed.

        antisym : bool | tuple of bool (default False)
            Whether to antisymmetrise the PAC results. If a tuple of bool, both
            forms of PAC are computed in turn.

        method : int | tuple of int (default ``1``)
            The method to use to compute TDE :footcite:`Nikias1988`. Can
            include ``[1, 2, 3, 4]``.

        n_jobs : int (default ``1``)
            The number of jobs to run in parallel. If ``-1``, all available
            CPUs are used.

        Notes
        -----
        TDE can be computed from the bispectrum, :math:`\textbf{B}`, of signals
        :math:`\textbf{x}` and :math:`\textbf{y}` of the seeds and targets,
        respectively, which has the general form

        :math:`\textbf{B}_{kmn}(f_1,f_2)=<\textbf{k}(f_1)\textbf{m}(f_2)
        \textbf{n}^*(f_2+f_1)>` ,

        where :math:`kmn` is a combination of signals with Fourier coefficients
        :math:`\textbf{k}`, :math:`\textbf{m}`, and :math:`\textbf{n}`,
        respectively; :math:`f_1` and :math:`f_2` correspond to a lower and
        higher frequency, respectively; and :math:`<>` represents the average
        value over epochs. When computing TDE, information from
        :math:`\textbf{n}` is taken not only from the positive frequencies, but
        also the negative frequencies.

        Four methods exist for computing TDE based on the bispectrum
        :footcite:`Nikias1988`. The fundamental equation is as follows

        :math:`\textrm{TDE}_{xy}(\tau)=\int_{-\pi}^{+\pi}\int_{-\pi}^{+\pi}
        \textbf{I}(\textbf{x}_{f_1},\textbf{y}_{f_2})e^{-if_1\tau}df_1df_2` ,

        where :math:`\textbf{I}` varies depending on the method; and
        :math:`\tau` is a given time delay. Phase information of the signals is
        extracted from the bispectrum in two variants used by the different
        methods:

        :math:`\boldsymbol{\phi}(\textbf{x}_{f_1},\textbf{y}_{f_2})=
        \boldsymbol{\varphi}_{\textbf{B}_{xyx}} (f_1,f_2)-\boldsymbol{
        \varphi}_{\textbf{B}_{xxx}}(f_1,f_2)` ;

        :math:`\boldsymbol{\phi}'(\textbf{x}_{f_1},\textbf{y}_{f_2})=
        \boldsymbol{\varphi}_{\textbf{B}_{xyx}}(f_1,f_2)-\frac{1}{2}(
        \boldsymbol{\varphi}_{\textbf{B}_{xxx}}(f_1,f_2) + \boldsymbol{
        \varphi}_{\textbf{B}_{yyy}}(f_1,f_2))` .

        **Method I**:
        :math:`\textbf{I}(\textbf{x}_{f_1},\textbf{y}_{f_2})=e^{i\boldsymbol{
        \phi}(\textbf{x}_{f_1},\textbf{y}_{f_2})}`

        **Method II**:
        :math:`\textbf{I}(\textbf{x}_{f_1},\textbf{y}_{f_2})=e^{i\boldsymbol{
        \phi}'(\textbf{x}_{f_1},\textbf{y}_{f_2})}`

        **Method III**:
        :math:`\textbf{I}(\textbf{x}_{f_1},\textbf{y}_{f_2})=\Large \frac{
        \textbf{B}_{xyx}(f_1,f_2)}{\textbf{B}_{xxx}(f_1,f_2)}`

        **Method IV**:
        :math:`\textbf{I}(\textbf{x}_{f_1},\textbf{y}_{f_2})=\Large \frac{
        |\textbf{B}_{xyx}(f_1,f_2)|e^{i\boldsymbol{\phi}'(\textbf{x}_{f_1},
        \textbf{y}_{f_2})}}{\sqrt{|\textbf{B}_{xxx}(f_1,f_2)||\textbf{B}_{yyy}
        (f_1,f_2)|}}`

        where :math:`\boldsymbol{\varphi}_{\textbf{B}}` is the phase of the
        bispectrum. All four methods aim to capture the phase difference
        between :math:`\textbf{x}` and :math:`\textbf{y}`. Method I involves
        the extraction of phase spectrum periodicity and monotony, with method
        III involving an additional amplitude weighting from the bispectrum of
        :math:`\textbf{x}`. Method II instead relies on a combination of phase
        spectra of the different frequency components, with method IV
        containing an additional amplitude weighting from the bispectrum of
        :math:`\textbf{x}` and :math:`\textbf{y}`. No single method is superior
        to another.

        Antisymmetrisation of the bispectrum is implemented as the replacement
        of :math:`\textbf{B}_{xyx}` with :math:`(\textbf{B}_{xxy} -
        \textbf{B}_{yxx})` in the above equations :footcite:`JurharInPrep`.

        If the seed and target for a given connection is the same channel, an
        error is raised.

        References
        ----------
        .. footbibliography::
        """
        self._reset_attrs()

        self._sort_metrics(antisym, method)
        self._sort_indices(indices)
        self._sort_parallelisation(n_jobs)

        if self.verbose:
            print("Computing TDE...\n")

        self._compute_bispectra()
        self._compute_tde()
        self._store_results()

        if self.verbose:
            print("    ... TDE computation finished\n")

    def _reset_attrs(self) -> None:
        """Reset attrs. of the object to prevent interference."""
        super()._reset_attrs()

        self._return_nosym = False
        self._return_antisym = False
        self._return_method_i = False
        self._return_method_ii = False
        self._return_method_iii = False
        self._return_method_iv = False

        self._bispectra = None

        self._tde_i_nosym = None
        self._tde_i_antisym = None
        self._tde_ii_nosym = None
        self._tde_ii_antisym = None
        self._tde_iii_nosym = None
        self._tde_iii_antisym = None
        self._tde_iv_nosym = None
        self._tde_iv_antisym = None

        self._xyz = None

    def _sort_metrics(
        self, antisym: bool | tuple[bool], method: int | tuple[int]
    ) -> None:
        """Sort inputs for the form of results being requested."""
        if not isinstance(antisym, (bool, tuple)):
            raise TypeError("`antisym` must be a bool or tuple of bools.")
        if not isinstance(method, (int, tuple)):
            raise TypeError("`method` must be an int or tuple of ints.")

        if isinstance(antisym, bool):
            antisym = (antisym,)
        if isinstance(method, int):
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
        indices = deepcopy(indices)
        if indices is None:
            indices = tuple(
                map(
                    tuple,
                    np.array(np.triu_indices(self._n_chans, 1)).tolist(),
                )
            )
        if not isinstance(indices, tuple):
            raise TypeError("`indices` must be a tuple.")
        if len(indices) != 2:
            raise ValueError("`indices` must have length of 2.")
        self._indices = deepcopy(indices)

        seeds = indices[0]
        targets = indices[1]
        for group_idcs in (seeds, targets):
            if not isinstance(group_idcs, tuple):
                raise TypeError("Entries of `indices` must be tuples.")
            if any(not isinstance(idx, int) for idx in group_idcs):
                raise TypeError(
                    "Entries for seeds and targets in `indices` must be ints."
                )
            if any(idx < 0 or idx >= self._n_chans for idx in group_idcs):
                raise ValueError(
                    "`indices` contains indices for channels not present in "
                    "the data."
                )
        if len(seeds) != len(targets):
            raise ValueError("Entries of `indices` must have equal length.")
        if any(seed == target for seed, target in zip(indices[0], indices[1])):
            raise ValueError(
                "Seeds and targets in `indices` must not be the same channel "
                "for any connection."
            )
        self._seeds = seeds
        self._targets = targets

        self._n_cons = len(seeds)

    def _compute_bispectra(self) -> None:
        """Compute bispectra between f1s and f2s of seeds and targets."""
        if self.verbose:
            print("    Computing bispectrum...")

        self._xyz = deepcopy(self._kmn)
        if not self._return_method_ii and not self._return_method_iv:
            del self._xyz["yyy"]
        if not self._return_nosym:
            del self._xyz["xyx"]
        if not self._return_antisym:
            del self._xyz["xxy"]
            del self._xyz["yxx"]

        hankel_freq_mask = hankel(
            np.arange(self._n_unique_freqs),
            np.arange(self._n_unique_freqs - 1, 2 * self._n_unique_freqs - 1),
        )
        args = [
            {
                "data": self.data[:, (seed, target)],
                "hankel_freq_mask": hankel_freq_mask,
                "kmn": tuple(self._xyz.values()),
            }
            for seed, target in zip(self._seeds, self._targets)
        ]

        # have to average complex values outside of Numba-compiled function
        self._bispectra = (
            np.array(
                pqdm(
                    args,
                    _compute_bispectrum_tde,
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
            print("        ... Bispectrum computation finished\n")

    def _compute_tde(self) -> None:
        """Compute TDE results from bispectra."""
        if self.verbose:
            print("    Computing TDE...")

        if self._return_nosym:
            self._compute_tde_nosym()

        if self._return_antisym:
            self._compute_tde_antisym()

        self._compute_times()

        if self.verbose:
            print("        ... TDE computation finished\n")

    def _compute_tde_nosym(self) -> None:
        """Compute unsymmetrised TDE."""
        B_xxx = self._bispectra[list(self._xyz.keys()).index("xxx")]

        if self._return_method_ii or self._return_method_iv:
            B_yyy = self._bispectra[list(self._xyz.keys()).index("yyy")]

        B_xyx = self._bispectra[list(self._xyz.keys()).index("xyx")]

        if self._return_method_i:
            self._tde_i_nosym = self._compute_tde_form_parallel(
                _compute_tde_i, {"B_xyx": B_xyx, "B_xxx": B_xxx}
            )
        if self._return_method_ii:
            self._tde_ii_nosym = self._compute_tde_form_parallel(
                _compute_tde_ii,
                {"B_xyx": B_xyx, "B_xxx": B_xxx, "B_yyy": B_yyy},
            )
        if self._return_method_iii:
            self._tde_iii_nosym = self._compute_tde_form_parallel(
                _compute_tde_iii, {"B_xyx": B_xyx, "B_xxx": B_xxx}
            )
        if self._return_method_iv:
            self._tde_iv_nosym = self._compute_tde_form_parallel(
                _compute_tde_iv,
                {"B_xyx": B_xyx, "B_xxx": B_xxx, "B_yyy": B_yyy},
            )

    def _compute_tde_antisym(self) -> None:
        """Compute antisymmetrised TDE."""
        B_xxx = self._bispectra[list(self._xyz.keys()).index("xxx")]

        if self._return_method_ii or self._return_method_iv:
            B_yyy = self._bispectra[list(self._xyz.keys()).index("yyy")]

        B_xyx = (
            self._bispectra[list(self._xyz.keys()).index("xxy")]
            - self._bispectra[list(self._xyz.keys()).index("yxx")]
        )

        if self._return_method_i:
            self._tde_i_antisym = self._compute_tde_form_parallel(
                _compute_tde_i, {"B_xyx": B_xyx, "B_xxx": B_xxx}
            )
        if self._return_method_ii:
            self._tde_ii_antisym = self._compute_tde_form_parallel(
                _compute_tde_ii,
                {"B_xyx": B_xyx, "B_xxx": B_xxx, "B_yyy": B_yyy},
            )
        if self._return_method_iii:
            self._tde_iii_antisym = self._compute_tde_form_parallel(
                _compute_tde_iii, {"B_xyx": B_xyx, "B_xxx": B_xxx}
            )
        if self._return_method_iv:
            self._tde_iv_antisym = self._compute_tde_form_parallel(
                _compute_tde_iv,
                {"B_xyx": B_xyx, "B_xxx": B_xxx, "B_yyy": B_yyy},
            )

    def _compute_tde_form_parallel(
        self, func: Callable, kwargs: dict
    ) -> np.ndarray:
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
            "PyBispectra Internal Error: `kwargs` passed to `pqdm` must be a "
            "dict. Please contact the PyBispectra developers."
        )

        con_kwargs = []
        for con_i in range(self._n_cons):
            con_kwargs.append(
                {key: value[con_i] for key, value in kwargs.items()}
            )

        return np.array(
            pqdm(
                con_kwargs,
                func,
                self._n_jobs,
                argument_type="kwargs",
                desc="Processing connections...",
                disable=not self.verbose,
            )
        )

    def _compute_times(self) -> None:
        """Compute timepoints (in ms) in the results."""
        epoch_dur = 0.5 * ((self.freqs.shape[0] - 1) / self.sampling_freq)
        self._times = (
            np.linspace(-epoch_dur, epoch_dur, self._n_freqs, dtype=np.float32)
            * 1000
        )  # use float32 to minimise rounding errors

    def _store_results(self) -> None:
        """Store computed results in objects."""
        results = []

        if self._tde_i_nosym is not None:
            results.append(
                ResultsTDE(
                    self._tde_i_nosym,
                    self._indices,
                    self._times,
                    "TDE | Method I",
                )
            )
        if self._tde_ii_nosym is not None:
            results.append(
                ResultsTDE(
                    self._tde_ii_nosym,
                    self._indices,
                    self._times,
                    "TDE | Method II",
                )
            )
        if self._tde_iii_nosym is not None:
            results.append(
                ResultsTDE(
                    self._tde_iii_nosym,
                    self._indices,
                    self._times,
                    "TDE | Method III",
                )
            )
        if self._tde_iv_nosym is not None:
            results.append(
                ResultsTDE(
                    self._tde_iv_nosym,
                    self._indices,
                    self._times,
                    "TDE | Method IV",
                )
            )

        if self._tde_i_antisym is not None:
            results.append(
                ResultsTDE(
                    self._tde_i_antisym,
                    self._indices,
                    self._times,
                    "TDE (antisymmetrised) | Method I",
                )
            )
        if self._tde_ii_antisym is not None:
            results.append(
                ResultsTDE(
                    self._tde_ii_antisym,
                    self._indices,
                    self._times,
                    "TDE (antisymmetrised) | Method II",
                )
            )
        if self._tde_iii_antisym is not None:
            results.append(
                ResultsTDE(
                    self._tde_iii_antisym,
                    self._indices,
                    self._times,
                    "TDE (antisymmetrised) | Method III",
                )
            )
        if self._tde_iv_antisym is not None:
            results.append(
                ResultsTDE(
                    self._tde_iv_antisym,
                    self._indices,
                    self._times,
                    "TDE (antisymmetrised) | Method IV",
                )
            )

        self._results = tuple(results)

    @property
    def results(self) -> ResultsTDE | tuple[ResultsTDE]:
        """Return the results.

        Returns
        -------
        results : ~pybispectra.utils.ResultsTDE | tuple of ~pybispectra.utils.ResultsTDE
            The results of the TDE computation returned as a single results
            object (if only one TDE variant was computed) or a tuple of results
            objects.
        """  # noqa: E501
        if len(self._results) == 1:
            return deepcopy(self._results[0])
        return deepcopy(self._results)


@njit
def _compute_bispectrum_tde(
    data: np.ndarray,
    hankel_freq_mask: np.ndarray,
    kmn: tuple[list[int]],
) -> np.ndarray:  # pragma: no cover
    """Compute the bispectrum for a single connection for use in TDE.

    Parameters
    ----------
    data : numpy.ndarray, shape of [epochs, 2, frequencies]
        Fourier coefficients, where the second dimension contains the data for
        the seed and target channel of a single connection, respectively.
        Contains coefficients for the zero frequency, the positive frequencies,
        and the negative frequencies, respectively.

    hankel_freq_mask : numpy.ndarray, shape of [frequencies, frequencies]
        Hankel matrix to use as a frequency mask for the frequencies in channel
        n of ``data``, where ``fs`` is the zero and positive frequencies.
        Can be generated with ``scipy.linalg.hankel(c=numpy.arange(0, fs),
        r=(numpy.arange(fs-1 : fs*2))``.

    kmn : tuple of list of int, shape of [x, 3]
        Tuple of variable length (x) of lists, where each list contains the k,
        m, and n channel indices in ``data``, respectively, to compute the
        bispectrum for.

    Returns
    -------
    results : numpy.ndarray, shape of [x, epochs, frequencies, frequencies]
        Complex-valued array containing the bispectrum of a single connection,
        where the first dimension corresponds to the different channel indices
        given in ``kmn``.

    Notes
    -----
    Averaging across epochs is not performed here as ``numpy.mean`` of
    complex numbers is not supported when compiling using Numba.

    No checks on the input data are performed for speed.
    """
    n_unique_freqs = hankel_freq_mask.shape[0]
    results = np.full(
        (len(kmn), data.shape[0], n_unique_freqs, n_unique_freqs),
        fill_value=np.nan,
        dtype=np.complex128,
    )

    for kmn_i, (k, m, n) in enumerate(kmn):
        for epoch_i, epoch_data in enumerate(data):
            # No arrays as indices in Numba, so loop over to pass int indices
            hankel_n = np.empty_like(hankel_freq_mask, dtype=np.complex128)
            for row_i in range(n_unique_freqs):
                for col_i in range(n_unique_freqs):
                    hankel_n[row_i, col_i] = epoch_data[
                        n, hankel_freq_mask[row_i, col_i]
                    ]

            results[kmn_i, epoch_i] = np.multiply(
                epoch_data[k, :n_unique_freqs],
                np.multiply(
                    epoch_data[m, :n_unique_freqs],
                    np.conjugate(hankel_n),
                ),
            )

    return results


def _compute_tde_i(B_xyx: np.ndarray, B_xxx: np.ndarray) -> np.ndarray:
    """Compute TDE from bispectra with method I for a single connection.

    Parameters
    ----------
    B_xyx : numpy.ndarray, shape of [frequencies, frequencies]
        Bispectrum for channel combination ``xyx``.

    B_xxx : numpy.ndarray, shape of [frequencies, frequencies]
        Bispectrum for channel combination ``xxx``.

    Returns
    -------
    tde : numpy.ndarray, shape of [times]
        Time delay estimates.

    Notes
    -----
    No checks on the input data are performed for speed.
    """
    I = np.zeros((B_xyx.shape[0] * 2 - 1), dtype=np.complex128)
    phi = np.angle(B_xyx) - np.angle(B_xxx)
    I[: B_xyx.shape[0]] = np.nansum(np.exp(1j * phi), axis=0)

    return np.abs(np.fft.fftshift(np.fft.ifft(I)))


def _compute_tde_ii(
    B_xyx: np.ndarray,
    B_xxx: np.ndarray,
    B_yyy: np.ndarray,
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

    Returns
    -------
    tde : numpy.ndarray, shape of [times]
        Time delay estimates.

    Notes
    -----
    No checks on the input data are performed for speed.
    """
    I = np.zeros((B_xyx.shape[0] * 2 - 1), dtype=np.complex128)
    phi_prime = np.angle(B_xyx) - 0.5 * (np.angle(B_xxx) + np.angle(B_yyy))
    I[: B_xyx.shape[0]] = np.nansum(np.exp(1j * phi_prime), axis=0)

    return np.abs(np.fft.fftshift(np.fft.ifft(I)))


def _compute_tde_iii(B_xyx: np.ndarray, B_xxx: np.ndarray) -> np.ndarray:
    """Compute TDE from bispectra with method III for a single connection.

    Parameters
    ----------
    B_xyx : numpy.ndarray, shape of [frequencies, frequencies]
        Bispectrum for channel combination ``xyx``.

    B_xxx : numpy.ndarray, shape of [frequencies, frequencies]
        Bispectrum for channel combination ``xxx``.

    Returns
    -------
    tde : numpy.ndarray, shape of [times]
        Time delay estimates.

    Notes
    -----
    No checks on the input data are performed for speed.
    """
    I = np.zeros((B_xyx.shape[0] * 2 - 1), dtype=np.complex128)
    I[: B_xyx.shape[0]] = np.nansum(np.divide(B_xyx, B_xxx), axis=0)

    return np.abs(np.fft.fftshift(np.fft.ifft(I)))


def _compute_tde_iv(
    B_xyx: np.ndarray,
    B_xxx: np.ndarray,
    B_yyy: np.ndarray,
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

    Returns
    -------
    tde : numpy.ndarray, shape of [times]
        Time delay estimates.

    Notes
    -----
    No checks on the input data are performed for speed.
    """
    I = np.zeros((B_xyx.shape[0] * 2 - 1), dtype=np.complex128)
    phi_prime = np.angle(B_xyx) - 0.5 * (np.angle(B_xxx) + np.angle(B_yyy))
    I[: B_xyx.shape[0]] = np.nansum(
        np.divide(
            np.multiply(np.abs(B_xyx), np.exp(1j * phi_prime)),
            np.sqrt(np.multiply(np.abs(B_xxx), np.abs(B_yyy))),
        ),
        axis=0,
    )

    return np.abs(np.fft.fftshift(np.fft.ifft(I)))
