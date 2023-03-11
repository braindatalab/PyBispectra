"""Tools for handling TDE analysis."""

import copy
from typing import Callable

import numpy as np
from numpy.typing import NDArray
from numba import njit
from pqdm.processes import pqdm

from pybispectra.utils import (
    _ProcessBispectra,
    ResultsTDE,
    _compute_bispectrum,
)


np.seterr(divide="ignore", invalid="ignore")  # no warning for NaN division


class TDE(_ProcessBispectra):
    """Class for computing time delay estimation (TDE) using bispectra.

    PARAMETERS
    ----------
    data : NumPy NDArray of float
    -   3D array of FFT coefficients with shape [epochs x channels x
        frequencies].

    freqs : NumPy NDArray of float
    -   1D array of the frequencies in `data`.

    verbose : bool; default True
    -   Whether or not to report the progress of the processing.

    METHODS
    -------
    compute
    -   Compute TDE, averaged over epochs.

    get_results
    -   Return a copy of the results.

    copy
    -   Return a copy of the object.

    ATTRIBUTES
    ----------
    data : NumPy NDArray of float
    -   FFT coefficients with shape [epochs x channels x frequencies].

    freqs : NumPy NDArray of float
    -   1D array of the frequencies in `data`.

    indices : tuple of NumPy NDArray of int
    -   2 arrays containing the seed and target indices (respectively) most
        recently used with `compute`.

    f1 : NumPy NDArray of float
    -   1D array of low frequencies most recently used with `compute`.

    f2 : NumPy NDArray of float
    -   1D array of high frequencies most recently used with `compute`.

    verbose : bool
    -   Whether or not to report the progress of the processing.

    REFERENCES
    ----------
    [1] Nikias & Pan (1988). Time Delay Estimation in Unknown Gaussian
        Spatially Correlated Noise. IEEE Transactions on Acoustics, Speech, and
        Signal Processing. DOI: 10.1109/29.9008.

    [2] Jurhar & Haufe (In Preparation). Estimating Signal Time-Delays under
        Mixed Noise Influence with Novel Cross- and Bispectrum Methods.
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

    def compute(
        self,
        indices: tuple[NDArray[np.int64]] | None = None,
        f1: NDArray[np.float64] | None = None,
        f2: NDArray[np.float64] | None = None,
        symmetrise: str | list[str] = ["none", "antisym"],
        method: int | list[int] = [1, 2, 3, 4],
        n_jobs: int = 1,
    ) -> None:
        """Compute TDE, averaged over epochs.

        PARAMETERS
        ----------
        indices: tuple of NumPy NDArray of int | None; default None
        -   Indices of the channels to compute TDE between. Should contain 2
            1D arrays of equal length for the seed and target indices,
            respectively. If None, coupling between all channels is computed.

        f1 : numpy NDArray of float | None; default None
        -   A 1D array of the lower frequencies to compute TDE on. If None, all
            frequencies are used.

        f2 : numpy NDArray of float | None; default None
        -   A 1D array of the higher frequencies to compute TDE on. If None,
            all frequencies are used.

        symmetrise : str | list of str; default ["none", "antisym"]
        -   Symmetrisation to perform when computing TDE. If "none", no
            symmetrisation is performed. If "antisym", antisymmetrisation is
            performed.

        method : int | list of int; default [1, 2, 3, 4]
        -   The method to use to compute TDE, as in [1].

        n_jobs : int; default 1
        -   The number of jobs to run in parallel.

        NOTES
        -----
        -   If the seed and target for a given connection is the same channel,
            NaN values are returned.
        -   TDE is computed between all values of `f1` and `f2`. If any value
            of `f1` is higher than `f2`, a NaN value is returned.
        """
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
            print("    [TDE computation finished]\n")

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
            symmetrise = [copy.copy(symmetrise)]
        if isinstance(method, int):
            method = [copy.copy(method)]

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

    def _compute_bispectra(self) -> None:
        """Compute bispectra between f1s and f2s of channels in `indices`."""
        if self.verbose:
            print("    Computing bispectra...")

        self._xyz = copy.deepcopy(self._kmn)
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
            print("        [Bispectra computation finished]\n")

    def _compute_tde(self) -> None:
        """Compute TDE results from bispectra."""
        if self.verbose:
            print("    Computing TDE...")

        if self._return_nosym:
            self._compute_tde_nosym()

        if self._return_antisym:
            self._compute_tde_antisym()

        if self.verbose:
            print("        [TDE computation finished]\n")

    def _compute_tde_nosym(self) -> None:
        """Compute unsymmetrised TDE."""
        B_xxx = self._bispectra[self._xyz.keys().index("xxx")]

        if self._return_method_ii or self._return_method_iv:
            B_yyy = self._bispectra[self._xyz.keys().index("yyy")]

        B_xyx = self._bispectra[self._xyz.keys().index("xyx")]

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
        B_xxx = self._bispectra[self._xyz.keys().index("xxx")]

        if self._return_method_ii or self._return_method_iv:
            B_yyy = self._bispectra[self._xyz.keys().index("yyy")]

        B_xyx = (
            self._bispectra[self._xyz.keys().index("xxy")]
            - self._bispectra[self._xyz.keys().index("yxx")]
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
    ) -> NDArray[np.float64]:
        """Compute TDE in parallel across connections for a single form.

        PAREMETERS
        ----------
        func : Callable
        -   TDE computation function to parallelise.

        kwargs : dict
        -   Arguments to pass to `func`.

        RETURNS
        -------
        tde : NumPy NDArray of float
        -   2D array of shape [connections x f2 * 2 - 1] containing the time
            delay estimates.
        """
        assert isinstance(kwargs, dict), (
            "PyBispectra Internal Error: `kwargs` passed to `pqdm` must be a "
            "dict. Please contact the PyBispectra developers."
        )

        return np.array(
            pqdm(
                kwargs,
                func,
                self._n_jobs,
                argument_type="kwargs",
                desc="Processing connections...",
                disable=not self.verbose,
            )
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

        if len(results) == 1:
            self._results = results[0]
        else:
            self._results = tuple(results)


def _compute_shift_ifft_I(I: NDArray[np.complex128]) -> NDArray[np.float64]:
    """Compute the zero-freq. center-shifted iFFT on the I matrix.

    PARAMETERS
    ----------
    I : NumPy NDArray of complex float
    -   Complex-valued 1D array of shape [f2 * 2 - 1] containing the bispectrum
        phase information for computing TDE, summed over the lower frequencies.

    RETURNS
    -------
    TDE : NumPy NDArray of float
    -   Real-valued 1D array of shape [f2 * 2 - 1] containing the time delay
        estimates.
    """
    return np.abs(np.fft.fftshift(np.fft.ifft(I)))


def _compute_tde_i(
    B_xyx: NDArray[np.complex128], B_xxx: NDArray[np.complex128]
) -> NDArray[np.float64]:
    """Compute TDE from bispectra with method I for a single connection.

    PARAMETERS
    ----------
    B_xyx : NumPy NDArray of complex float
    -   2D array of shape [f1 x f2] containing the bispectrum for channel
        combination xyx.

    B_xxx : NumPy NDArray of complex float
    -   2D array of shape [f1 x f2] containing the bispectrum for channel
        combination xxx.

    RETURNS
    -------
    tde : NumPy NDArray of float
    -   1D array of shape [f2 * 2 - 1] containing the time delay estimates.

    NOTES
    -----
    -   No checks on the input data are performed for speed.
    """
    I = np.zeros((B_xyx.shape[0], B_xyx.shape[1] * 2 - 1), dtype=np.complex128)
    phi = np.angle(B_xyx) - np.angle(B_xxx)
    I[:, : B_xyx.shape[1]] = np.exp(1j * phi)

    return _compute_shift_ifft_I(np.sum(I, axis=1))


def _compute_tde_ii(
    B_xyx: NDArray[np.complex128],
    B_xxx: NDArray[np.complex128],
    B_yyy: NDArray[np.complex128],
) -> NDArray[np.float64]:
    """Compute TDE from bispectra with method II for a single connection.

    PARAMETERS
    ----------
    B_xyx : NumPy NDArray of complex float
    -   2D array of shape [f1 x f2] containing the bispectrum for channel
        combination xyx.

    B_xxx : NumPy NDArray of complex float
    -   2D array of shape [f1 x f2] containing the bispectrum for channel
        combination xxx.

    B_yyy : NumPy NDArray of complex float
    -   2D array of shape [f1 x f2] containing the bispectrum for channel
        combination yyy.

    RETURNS
    -------
    tde : NumPy NDArray of float
    -   1D array of shape [f2 * 2 - 1] containing the time delay estimates.

    NOTES
    -----
    -   No checks on the input data are performed for speed.
    """
    I = np.zeros((B_xyx.shape[0], B_xyx.shape[1] * 2 - 1), dtype=np.complex128)
    phi = np.angle(B_xyx) - 0.5 * (np.angle(B_xxx) - np.angle(B_yyy))
    I[:, : B_xyx.shape[1]] = np.exp(1j * phi)

    return _compute_shift_ifft_I(np.sum(I, axis=1))


def _compute_tde_iii(
    B_xyx: NDArray[np.complex128], B_xxx: NDArray[np.complex128]
) -> NDArray[np.float64]:
    """Compute TDE from bispectra with method III for a single connection.

    PARAMETERS
    ----------
    B_xyx : NumPy NDArray of complex float
    -   2D array of shape [f1 x f2] containing the bispectrum for channel
        combination xyx.

    B_xxx : NumPy NDArray of complex float
    -   2D array of shape [f1 x f2] containing the bispectrum for channel
        combination xxx.

    RETURNS
    -------
    tde : NumPy NDArray of float
    -   1D array of shape [f2 * 2 - 1] containing the time delay estimates.

    NOTES
    -----
    -   No checks on the input data are performed for speed.
    """
    I = np.zeros((B_xyx.shape[0], B_xyx.shape[1] * 2 - 1), dtype=np.complex128)
    I[:, : B_xyx.shape[1]] = B_xyx / B_xxx

    return _compute_shift_ifft_I(np.sum(I, axis=1))


def _compute_tde_iv(
    B_xyx: NDArray[np.complex128],
    B_xxx: NDArray[np.complex128],
    B_yyy: NDArray[np.complex128],
) -> NDArray[np.float64]:
    """Compute TDE from bispectra with method IV for a single connection.

    PARAMETERS
    ----------
    B_xyx : NumPy NDArray of complex float
    -   2D array of shape [f1 x f2] containing the bispectrum for channel
        combination xyx.

    B_xxx : NumPy NDArray of complex float
    -   2D array of shape [f1 x f2] containing the bispectrum for channel
        combination xxx.

    B_yyy : NumPy NDArray of complex float
    -   2D array of shape [f1 x f2] containing the bispectrum for channel
        combination yyy.

    RETURNS
    -------
    tde : NumPy NDArray of float
    -   1D array of shape [f2 * 2 - 1] containing the time delay estimates.

    NOTES
    -----
    -   No checks on the input data are performed for speed.
    """
    I = np.zeros((B_xyx.shape[0], B_xyx.shape[1] * 2 - 1), dtype=np.complex128)
    phi = np.angle(B_xyx) - 0.5 * (np.angle(B_xxx) - np.angle(B_yyy))
    I[:, : B_xyx.shape[1]] = (
        np.abs(B_xyx)
        * np.exp(1j * phi)
        / np.sqrt(np.abs(B_xxx) * np.abs(B_yyy))
    )

    return _compute_shift_ifft_I(np.sum(I, axis=1))
