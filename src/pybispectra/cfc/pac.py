"""Tools for handling PAC analysis."""

import copy

import numpy as np
from numba import njit
from pqdm.processes import pqdm

from pybispectra.utils import (
    _ProcessBispectra,
    ResultsCFC,
    _compute_bispectrum,
    fast_find_first,
)


np.seterr(divide="ignore", invalid="ignore")  # no warning for NaN division


class PAC(_ProcessBispectra):
    """Class for computing phase-amplitude coupling (PAC) using bispectra.

    Parameters
    ----------
    data : numpy.ndarray of float
        3D array of FFT coefficients with shape `[epochs x channels x
        frequencies]`.

    freqs : numpy.ndarray of float
        1D array of the frequencies in :attr:`data`.

    verbose : bool (default True)
        Whether or not to report the progress of the processing.

    Attributes
    ----------
    results : tuple of ResultsCFC
        PAC results for each of the computed metrics.

    data : numpy.ndarray of float
        FFT coefficients with shape `[epochs x channels x frequencies]`.

    freqs : numpy.ndarray of float
        1D array of the frequencies in :attr:`data`.

    indices : tuple of numpy.ndarray of int
        Two arrays containing the seed and target indices (respectively) most
        recently used with :meth:`compute`.

    f1 : numpy.ndarray of float
        1D array of low frequencies most recently used with :meth:`compute`.

    f2 : numpy.ndarray of float
        1D array of high frequencies most recently used with :meth:`compute`.

    verbose : bool
        Whether or not to report the progress of the processing.
    """

    _return_nosym = False
    _return_antisym = False
    _return_nonorm = False
    _return_threenorm = False

    _bispectra = None
    _bicoherence = None

    _pac_nosym_nonorm = None
    _pac_nosym_threenorm = None
    _pac_antisym_nonorm = None
    _pac_antisym_threenorm = None

    def compute(
        self,
        indices: tuple[np.ndarray] | None = None,
        f1: np.ndarray | None = None,
        f2: np.ndarray | None = None,
        symmetrise: str | list[str] = ["none", "antisym"],
        normalise: str | list[str] = ["none", "threenorm"],
        n_jobs: int = 1,
    ) -> None:
        """Compute PAC, averaged over epochs.

        Parameters
        ----------
        indices : tuple of numpy.ndarray of int | None (default None)
            Indices of the channels to compute PAC between. Should contain two
            1D arrays of equal length for the seed and target indices,
            respectively. If ``None``, coupling between all channels is
            computed.

        f1 : numpy.ndarray of float | None (default None)
            A 1D array of the lower frequencies to compute PAC on. If ``None``,
            all frequencies are used.

        f2 : numpy.ndarray of float | None (default None)
            A 1D array of the higher frequencies to compute PAC on. If
            ``None``, all frequencies are used.

        symmetrise : str | list of str (default ``["none", "antisym"]``)
            Symmetrisation to perform when computing PAC. If ``"none"``, no
            symmetrisation is performed. If ``"antisym"``, antisymmetrisation
            is performed.

        normalise : str | list of str (default ``["none", "threenorm"]``)
            Normalisation to perform when computing PAC. If ``"none"``, no
            normalisation is performed. If ``"threenorm"``, the bispectra is
            normalised to the bicoherence using a threenorm.

        n_jobs : int (default ``1``)
            The number of jobs to run in parallel.

        Notes
        -----
        -   If the seed and target for a given connection is the same channel
            and antisymmetrisation is being performed, ``numpy.nan`` values are
            returned.
        -   PAC is computed between all values of :attr:`f1` and :attr:`f2`. If
            any value of :attr:`f1` is higher than :attr:`f2`, a ``numpy.nan``
            value is returned.
        """
        self._reset_attrs()

        self._sort_metrics(symmetrise, normalise)
        self._sort_indices(indices)
        self._sort_freqs(f1, f2)
        self._sort_parallelisation(n_jobs)

        if self.verbose:
            print("Computing PAC...\n")

        self._compute_bispectra()
        if self._return_threenorm:
            self._compute_bicoherence()
        self._compute_pac()
        self._store_results()

        if self.verbose:
            print("    [PAC computation finished]\n")

    def _reset_attrs(self) -> None:
        """Reset attrs. of the object to prevent interference."""
        super()._reset_attrs()

        self._return_nosym = False
        self._return_antisym = False
        self._return_nonorm = False
        self._return_threenorm = False

        self._bispectra = None
        self._bicoherence = None

        self._pac_nosym_nonorm = None
        self._pac_nosym_threenorm = None
        self._pac_antisym_nonorm = None
        self._pac_antisym_threenorm = None

    def _sort_metrics(
        self, symmetrise: str | list[str], normalise: str | list[str]
    ) -> None:
        """Sort inputs for the form of results being requested."""
        if not isinstance(symmetrise, str) and not isinstance(
            symmetrise, list
        ):
            raise TypeError(
                "`symmetrise` must be a list of strings or a string."
            )
        if not isinstance(normalise, str) and not isinstance(normalise, list):
            raise TypeError(
                "`normalise` must be a list of strings or a string."
            )

        if isinstance(symmetrise, str):
            symmetrise = [copy.copy(symmetrise)]
        if isinstance(normalise, str):
            normalise = [copy.copy(normalise)]

        supported_sym = ["none", "antisym"]
        if any(entry not in supported_sym for entry in symmetrise):
            raise ValueError("The value of `symmetrise` is not recognised.")
        supported_norm = ["none", "threenorm"]
        if any(entry not in supported_norm for entry in normalise):
            raise ValueError("The value of `normalise` is not recognised.")

        if "none" in symmetrise:
            self._return_nosym = True
        if "antisym" in symmetrise:
            self._return_antisym = True

        if "none" in normalise:
            self._return_nonorm = True
        if "threenorm" in normalise:
            self._return_threenorm = True

    def _compute_bispectra(self) -> None:
        """Compute bispectra between f1s and f2s of seeds and targets."""
        if self.verbose:
            print("    Computing bispectra...")

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
                "f1s": self.f1,
                "f2s": self.f2,
                "kmn": kmn,
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

    def _compute_bicoherence(self) -> None:
        """Compute bicoherence from the bispectra using the threenorm."""
        if self.verbose:
            print("    Computing bicoherence...")

        args = [
            {
                "data": self.data[:, (seed, target)],
                "freqs": self.freqs,
                "f1s": self.f1,
                "f2s": self.f2,
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
            )
        )

        self._bicoherence = self._bispectra / threenorm

        if self.verbose:
            print("        [Bicoherence computation finished]\n")

    def _compute_pac(self) -> None:
        """Compute PAC results from bispectra/bicoherence."""
        if self._return_nonorm:
            if self._return_nosym:
                self._pac_nosym_nonorm = np.abs(self._bispectra[0])
            if self._return_antisym:
                self._pac_antisym_nonorm = np.abs(
                    self._bispectra[0] - self._bispectra[1]
                )
                for con_i, seed, target in zip(
                    range(self._n_cons), self._seeds, self._targets
                ):
                    if seed == target:
                        self._pac_antisym_nonorm[con_i] = np.full_like(
                            self._pac_antisym_nonorm[con_i], fill_value=np.nan
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
                        )

    def _store_results(self) -> None:
        """Store computed results in objects."""
        results = []

        if self._pac_nosym_nonorm is not None:
            results.append(
                ResultsCFC(
                    self._pac_nosym_nonorm,
                    self.indices,
                    self.f1,
                    self.f2,
                    "PAC - Unsymmetrised Bispectra",
                )
            )

        if self._pac_nosym_threenorm is not None:
            results.append(
                ResultsCFC(
                    self._pac_nosym_threenorm,
                    self.indices,
                    self.f1,
                    self.f2,
                    "PAC - Unsymmetrised Bicoherence",
                )
            )

        if self._pac_antisym_nonorm is not None:
            results.append(
                ResultsCFC(
                    self._pac_antisym_nonorm,
                    self.indices,
                    self.f1,
                    self.f2,
                    "PAC - Antisymmetrised Bispectra",
                )
            )

        if self._pac_antisym_threenorm is not None:
            results.append(
                ResultsCFC(
                    self._pac_antisym_threenorm,
                    self.indices,
                    self.f1,
                    self.f2,
                    "PAC - Antisymmetrised Bicoherence",
                )
            )

        self._results = tuple(results)

    @property
    def results(self) -> tuple[ResultsCFC]:
        """Return the results."""
        return self._results


@njit
def _compute_threenorm(
    data: np.ndarray,
    freqs: np.ndarray,
    f1s: np.ndarray,
    f2s: np.ndarray,
) -> np.ndarray:
    """Compute threenorm for a single connection across epochs.

    PARAMETERS
    ----------
    data : numpy.ndarray of float
        3D array of FFT coefficients with shape `[epochs x 2 x frequencies]`,
        where the second dimension contains the data for the seed and target
        channel of a single connection, respectively.

    freqs : numpy.ndarray of float
        1D array of frequencies in ``data``.

    f1s : numpy.ndarray of float
        1D array of low frequencies to compute the threenorm for.

    f2s : numpy.ndarray of float
        1D array of high frequencies to compute the threenorm for.

    RETURNS
    -------
    results : numpy.ndarray of float
        2D array containing the threenorm of a single connection averaged
        across epochs, with shape `[f1 x f2]`.
    """
    results = np.full(
        (f1s.shape[0], f2s.shape[0]), fill_value=np.nan, dtype=np.float64
    )
    for f1_i, f1 in enumerate(f1s):
        for f2_i, f2 in enumerate(f2s):
            if f1 < f2 and (f2 + f1) in freqs:
                fft_f1 = data[:, 0, fast_find_first(freqs, f1)]
                fft_f2 = data[:, 1, fast_find_first(freqs, f2)]
                fft_fdiff = data[:, 1, fast_find_first(freqs, f2 + f1)]
                results[f1_i, f2_i] = (
                    (np.abs(fft_f1) ** 3).mean()
                    * (np.abs(fft_f2) ** 3).mean()
                    * (np.abs(fft_fdiff) ** 3).mean()
                ) ** 1 / 3

    return results
