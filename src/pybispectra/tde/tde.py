"""Tools for handling TDE analysis."""

from copy import deepcopy
from typing import Callable

import numpy as np
from pqdm.processes import pqdm

from pybispectra.utils import ResultsTDE
from pybispectra.utils._process import _ProcessBispectrum, _compute_bispectrum


class TDE(_ProcessBispectrum):
    """Class for computing time delay estimation (TDE) using bispectra.

    Parameters
    ----------
    data : numpy.ndarray of float, shape of [epochs, channels, frequencies]
        FFT coefficients.

    freqs : numpy.ndarray of float, shape of [frequencies]
        Frequencies (in Hz) in :attr:`data`.

    sfreq : float
        Sampling frequency (in Hz) of the data from which :attr:`data` was
        derived.

    verbose : bool (default True)
        Whether or not to report the progress of the processing.

    Attributes
    ----------
    results : tuple of ResultsTDE
        TDE results for each of the computed metrics.

    data : numpy.ndarray of float, shape of [epochs, channels, frequencies]
        FFT coefficients.

    freqs : numpy.ndarray of float, shape of [frequencies]
        Frequencies (in Hz) in :attr:`data`.

    sfreq : float
        Sampling frequency (in Hz) of the data from which :attr:`data` was
        derived.

    indices : tuple of numpy.ndarray of int, length of 2
        Indices of the seed and target channels, respectively, most recently
        used with :meth:`compute`.

    f1 : numpy.ndarray of float, shape of [frequencies]
        Low frequencies (in Hz) most recently used with :meth:`compute`.

    f2 : numpy.ndarray of float, shape of [frequencies]
        High frequencies (in Hz) most recently used with :meth:`compute`.

    verbose : bool
        Whether or not to report the progress of the processing.
    """

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
        sfreq: int | float,
        verbose: bool = True,
    ) -> None:  # noqa D107
        super().__init__(data, freqs, verbose)
        self._sort_sfreq(sfreq)

    def _sort_sfreq(self, sfreq: int | float):
        """Sort sampling frequency input."""
        if not isinstance(sfreq, float) and not isinstance(sfreq, int):
            raise TypeError("`sfreq` must be an int or a float.")

        self.sfreq = deepcopy(sfreq)

    def compute(
        self,
        indices: tuple[np.ndarray] | None = None,
        f1: np.ndarray | None = None,
        f2: np.ndarray | None = None,
        symmetrise: str | list[str] = ["none", "antisym"],
        method: int | list[int] = [1, 2, 3, 4],
        n_jobs: int = 1,
    ) -> None:
        r"""Compute TDE, averaged over epochs.

        Parameters
        ----------
        indices : tuple of numpy.ndarray of int | None (default None), length of 2
            Indices of the seed and target channels, respectively, to compute
            TDE between. If ``None``, coupling between all channels is
            computed.

        f1 : numpy.ndarray of float | None (default None), shape of [frequencies]
            Lower frequencies to compute TDE on. If ``None``, all frequencies
            are used.

        f2 : numpy.ndarray of float | None (default None), shape of [frequencies]
            Higher frequencies to compute TDE on. If ``None``, all frequencies
            are used.

        symmetrise : str | list of str (default ``["none", "antisym"]``)
            Symmetrisation to perform when computing TDE. If "none", no
            symmetrisation is performed. If "antisym", antisymmetrisation is
            performed.

        method : int | list of int (default ``[1, 2, 3, 4]``)
            The method to use to compute TDE :footcite:`Nikias1988`.

        n_jobs : int (default ``1``)
            The number of jobs to run in parallel.

        Notes
        -----
        TDE can be computed from the bispectrum, :math:`B`, of signals
        :math:`\vec{x}` and :math:`\vec{y}` of the seeds and targets,
        respectively, which has the general form:

        :math:`\large B_{kmn}(f_1,f_2)=<\vec{k}(f_1)\vec{m}(f_2)\vec{n}^*(f_2+f_1)>`,

        where :math:`kmn` is a combination of channels :math:`\vec{x}` and
        :math:`\vec{y}`, and the angled brackets represent the averaged value
        over epochs. Four methods exist for computing TDE based on the
        bispectrum :footcite:`Nikias1988`. The fundamental equation is as
        follows:

        :math:`\large TDE_{xy}(\tau)=\int_{-\pi}^{+\pi}\int_{-\pi}^{+\pi}I(\vec{x}_{f_1},\vec{y}_{f_2})e^{-if_1\tau}df_1df_2`,

        where :math:`I` varies depending on the method, and :math:`\tau` is a
        given time delay. Phase information of the signals is extracted from
        the bispectrum in two variants used by the different methods:

        :math:`\large \phi(\vec{x}_{f_1},\vec{y}_{f_2})=\varphi_{B_{xyx}}(f_1,f_2)-\varphi_{B_{xxx}}(f_1,f_2)`

        :math:`\large \phi'(\vec{x}_{f_1},\vec{y}_{f_2})=\varphi_{B_{xyx}}(f_1,f_2)-\frac{1}{2}(\varphi_{B_{xxx}}(f_1, f_2) + \varphi_{B_{yyy}}(f_1,f_2))`

        **Method I**:
        :math:`\large I(\vec{x}_{f_1},\vec{y}_{f_2})=e^{i\phi(\vec{x}_{f_1},\vec{y}_{f_2})}`

        **Method II**:
        :math:`\large I(\vec{x}_{f_1},\vec{y}_{f_2})=e^{i\phi'(\vec{x}_{f_1},\vec{y}_{f_2})}`

        **Method III**:
        :math:`\large I(\vec{x}_{f_1},\vec{y}_{f_2})=\Large \frac{B_{xyx}(f_1,f_2)}{B_{xxx}(f_1,f_2)}`

        **Method IV**:
        :math:`\large I(\vec{x}_{f_1},\vec{y}_{f_2})=\Large \frac{|B_{xyx}(f_1,f_2)|e^{i\phi'(\vec{x}_{f_1},\vec{y}_{f_2})}}{\sqrt{|B_{xxx}(f_1,f_2)||B_{yyy}(f_1,f_2)|}}`

        where :math:`\varphi_{B}` is the phase of the bispectrum.
        Antisymmetrisation of the bispectrum is implemented as the replacement
        of :math:`B_{xyx}` with :math:`(B_{xxy} - B_{yxx})` in the above
        equations :footcite:`JurharInPrep`.

        If the seed and target for a given connection is the same channel, an
        error is raised.

        TDE is computed between all values of :attr:`f1` and :attr:`f2`. If any
        value of :attr:`f1` is higher than :attr:`f2`, a ``numpy.nan`` value is
        returned.

        References
        ----------
        .. footbibliography::
        """  # noqa E501
        self._reset_attrs()

        self._sort_metrics(symmetrise, method)
        self._sort_indices(indices)
        self._sort_freqs(f1, f2)
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
        self, symmetrise: str | list[str], method: int | list[int]
    ) -> None:
        """Sort inputs for the form of results being requested."""
        if not isinstance(symmetrise, str) and not isinstance(
            symmetrise, list
        ):
            raise TypeError(
                "`symmetrise` must be a list of strings or a string."
            )
        if not isinstance(method, int) and not isinstance(method, list):
            raise TypeError("`method` must be a list of ints or an int.")

        if isinstance(symmetrise, str):
            symmetrise = [deepcopy(symmetrise)]
        if isinstance(method, int):
            method = [deepcopy(method)]

        supported_sym = ["none", "antisym"]
        if any(entry not in supported_sym for entry in symmetrise):
            raise ValueError("The value of `symmetrise` is not recognised.")
        supported_meth = [1, 2, 3, 4]
        if any(entry not in supported_meth for entry in method):
            raise ValueError("The value of `method` is not recognised.")

        if "none" in symmetrise:
            self._return_nosym = True
        if "antisym" in symmetrise:
            self._return_antisym = True

        if 1 in method:
            self._return_method_i = True
        if 2 in method:
            self._return_method_ii = True
        if 3 in method:
            self._return_method_iii = True
        if 4 in method:
            self._return_method_iv = True

    def _sort_indices(self, indices: np.ndarray | None) -> None:
        """Sort seed-target indices inputs."""
        indices = deepcopy(indices)
        if indices is None:
            seed_indices = np.reshape(
                np.repeat(range(self._n_chans), self._n_chans),
                (self._n_chans, self._n_chans),
            )
            target_indices = np.reshape(
                np.tile(range(self._n_chans), self._n_chans),
                (self._n_chans, self._n_chans),
            )
            indices = (
                seed_indices[np.triu_indices(self._n_chans, 1)],
                target_indices[np.triu_indices(self._n_chans, 1)],
            )
        if not isinstance(indices, tuple):
            raise TypeError("`indices` should be a tuple.")
        if len(indices) != 2:
            raise ValueError("`indices` should have a length of 2.")
        self.indices = deepcopy(indices)

        seeds = indices[0]
        targets = indices[1]
        for group_idcs in (seeds, targets):
            if not isinstance(group_idcs, np.ndarray):
                raise TypeError("Entries of `indices` should be NumPy arrays.")
            if any(idx < 0 or idx >= self._n_chans for idx in group_idcs):
                raise ValueError(
                    "`indices` contains indices for channels not present in "
                    "the data."
                )
        if len(seeds) != len(targets):
            raise ValueError("Entires of `indices` must have equal length.")
        if any(seed == target for seed, target in zip(indices[0], indices[1])):
            raise ValueError(
                "Seeds and targets in `indices` should not be the same "
                "channel for any connection."
            )
        self._seeds = seeds
        self._targets = targets

        self._n_cons = len(seeds)

    def _compute_bispectra(self) -> None:
        """Compute bispectra between f1s and f2s of seeds and targets."""
        if self.verbose:
            print("    Computing bispectra...")

        self._xyz = deepcopy(self._kmn)
        if not self._return_method_ii and not self._return_method_iv:
            del self._xyz["yyy"]
        if not self._return_nosym:
            del self._xyz["xyx"]
        if not self._return_antisym:
            del self._xyz["xxy"]
            del self._xyz["yxx"]

        args = [
            {
                "data": self.data[:, (seed, target)],
                "freqs": self.freqs,
                "f1s": self.f1,
                "f2s": self.f2,
                "kmn": tuple(self._xyz.values()),
            }
            for seed, target in zip(self._seeds, self._targets)
        ]

        # have to average complex value outside of Numba-compiled function
        self._bispectra = (
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
            print("        ... Bispectra computation finished\n")

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
        tde : numpy.ndarray of float, shape of [connections, f2 * 2 - 1]
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
        n_samples = len(self.f2)
        half_epoch_len = (n_samples / self.sfreq) * 0.5

        self._times = (
            np.linspace(-half_epoch_len, half_epoch_len, n_samples * 2 + 1)
            * 1000
        )

    def _store_results(self) -> None:
        """Store computed results in objects."""
        results = []

        if self._tde_i_nosym is not None:
            results.append(
                ResultsTDE(
                    self._tde_i_nosym,
                    self.indices,
                    self._times,
                    "Unsymmetrised TDE | Method I",
                )
            )
        if self._tde_ii_nosym is not None:
            results.append(
                ResultsTDE(
                    self._tde_ii_nosym,
                    self.indices,
                    self._times,
                    "Unsymmetrised TDE | Method II",
                )
            )
        if self._tde_iii_nosym is not None:
            results.append(
                ResultsTDE(
                    self._tde_iii_nosym,
                    self.indices,
                    self._times,
                    "Unsymmetrised TDE | Method III",
                )
            )
        if self._tde_iv_nosym is not None:
            results.append(
                ResultsTDE(
                    self._tde_iv_nosym,
                    self.indices,
                    self._times,
                    "Unsymmetrised TDE | Method IV",
                )
            )

        if self._tde_i_antisym is not None:
            results.append(
                ResultsTDE(
                    self._tde_i_antisym,
                    self.indices,
                    self._times,
                    "Antisymmetrised TDE | Method I",
                )
            )
        if self._tde_ii_antisym is not None:
            results.append(
                ResultsTDE(
                    self._tde_ii_antisym,
                    self.indices,
                    self._times,
                    "Antisymmetrised TDE | Method II",
                )
            )
        if self._tde_iii_antisym is not None:
            results.append(
                ResultsTDE(
                    self._tde_iii_antisym,
                    self.indices,
                    self._times,
                    "Antisymmetrised TDE | Method III",
                )
            )
        if self._tde_iv_antisym is not None:
            results.append(
                ResultsTDE(
                    self._tde_iv_antisym,
                    self.indices,
                    self._times,
                    "Antisymmetrised TDE | Method IV",
                )
            )

        self._results = tuple(results)

    @property
    def results(self) -> tuple[ResultsTDE]:
        """Return the results."""
        return self._results


def _compute_shift_ifft_I(I: np.ndarray) -> np.ndarray:
    """Compute the zero-freq. center-shifted iFFT on the ``I`` matrix.

    PARAMETERS
    ----------
    I : numpy.ndarray of complex float, shape of [f2 * 2]
        Bispectrum phase information for computing TDE, summed over the lower
        frequencies.

    RETURNS
    -------
    tde : numpy.ndarray of float, shape of [f2 * 2]
        Time delay estimates.
    """
    return np.abs(np.fft.fftshift(np.fft.ifft(I)))


def _compute_tde_i(B_xyx: np.ndarray, B_xxx: np.ndarray) -> np.ndarray:
    """Compute TDE from bispectra with method I for a single connection.

    Parameters
    ----------
    B_xyx : numpy.ndarray of complex float, shape of [f1s, f2s]
        Bispectrum for channel combination `xyx`.

    B_xxx : numpy.ndarray of complex float, shape of [f1s, f2s]
        Bispectrum for channel combination `xxx`.

    Returns
    -------
    tde : numpy.ndarray of float, shape of [f2 * 2]
        Time delay estimates.

    Notes
    -----
    No checks on the input data are performed for speed.
    """
    I = np.zeros((B_xyx.shape[1] * 2 + 1), dtype=np.complex128)
    phi = np.angle(B_xyx) - np.angle(B_xxx)
    I[: B_xyx.shape[1]] = np.nansum(np.exp(1j * phi), axis=0)

    return _compute_shift_ifft_I(I)


def _compute_tde_ii(
    B_xyx: np.ndarray,
    B_xxx: np.ndarray,
    B_yyy: np.ndarray,
) -> np.ndarray:
    """Compute TDE from bispectra with method II for a single connection.

    Parameters
    ----------
    B_xyx : numpy.ndarray of complex float, shape of [f1s, f2s]
        Bispectrum for channel combination `xyx`.

    B_xxx : numpy.ndarray of complex float, shape of [f1s, f2s]
        Bispectrum for channel combination `xxx`.

    B_yyy : numpy.ndarray of complex float, shape of [f1s, f2s]
        Bispectrum for channel combination `yyy`.

    Returns
    -------
    tde : numpy.ndarray of float, shape of [f2 * 2]
        Time delay estimates.

    Notes
    -----
    No checks on the input data are performed for speed.
    """
    I = np.zeros((B_xyx.shape[1] * 2 + 1), dtype=np.complex128)
    phi_prime = np.angle(B_xyx) - 0.5 * (np.angle(B_xxx) + np.angle(B_yyy))
    I[: B_xyx.shape[1]] = np.nansum(np.exp(1j * phi_prime), axis=0)

    return _compute_shift_ifft_I(I)


def _compute_tde_iii(B_xyx: np.ndarray, B_xxx: np.ndarray) -> np.ndarray:
    """Compute TDE from bispectra with method III for a single connection.

    Parameters
    ----------
    B_xyx : numpy.ndarray of complex float, shape of [f1s, f2s]
        Bispectrum for channel combination `xyx`.

    B_xxx : numpy.ndarray of complex float, shape of [f1s, f2s]
        Bispectrum for channel combination `xxx`.

    Returns
    -------
    tde : numpy.ndarray of float, shape of [f2 * 2]
        Time delay estimates.

    Notes
    -----
    No checks on the input data are performed for speed.
    """
    I = np.zeros((B_xyx.shape[1] * 2 + 1), dtype=np.complex128)
    I[: B_xyx.shape[1]] = np.nansum(np.divide(B_xyx, B_xxx), axis=0)

    return _compute_shift_ifft_I(I)


def _compute_tde_iv(
    B_xyx: np.ndarray,
    B_xxx: np.ndarray,
    B_yyy: np.ndarray,
) -> np.ndarray:
    """Compute TDE from bispectra with method IV for a single connection.

    Parameters
    ----------
    B_xyx : numpy.ndarray of complex float, shape of [f1s, f2s]
        Bispectrum for channel combination `xyx`.

    B_xxx : numpy.ndarray of complex float, shape of [f1s, f2s]
        Bispectrum for channel combination `xxx`.

    B_yyy : numpy.ndarray of complex float, shape of [f1s, f2s]
        Bispectrum for channel combination `yyy`.

    Returns
    -------
    tde : numpy.ndarray of float, shape of [f2 * 2]
        Time delay estimates.

    Notes
    -----
    No checks on the input data are performed for speed.
    """
    I = np.zeros((B_xyx.shape[1] * 2 + 1), dtype=np.complex128)
    phi_prime = np.angle(B_xyx) - 0.5 * (np.angle(B_xxx) + np.angle(B_yyy))
    I[: B_xyx.shape[1]] = np.nansum(
        np.divide(
            np.multiply(np.abs(B_xyx), np.exp(1j * phi_prime)),
            np.sqrt(np.multiply(np.abs(B_xxx), np.abs(B_yyy))),
        ),
        axis=0,
    )

    return _compute_shift_ifft_I(I)
