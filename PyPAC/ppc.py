"""Tools for handling PPC analysis."""

import numpy as np
from numba import njit
from pqdm.processes import pqdm

from process import Process


class PPC(Process):
    """Class for computing phase-phase coupling (PPC).

    PARAMETERS
    ----------
    data : NumPy ndarray
    -   FFT coefficients with shape [epochs x channels x frequencies].

    freqs : NumPy ndarray
    -   1D array of the frequencies in `data`.

    verbose : bool; default True
    -   Whether or not to report the progress of the processing.

    METHODS
    -------
    compute
    -   Compute PPC, averagee over epochs.

    get_results
    -   Return a copy of the results.

    copy
    -   Return a copy of the object.

    ATTRIBUTES
    ----------
    data : NumPy ndarray
    -   FFT coefficients with shape [epochs x channels x frequencies].

    freqs : NumPy ndarray
    -   1D array of the frequencies in `data`.

    indices : tuple of NumPy ndarray
    -   2 arrays containing the seed and target indices (respectively) most
        recently used with `compute`.

    f1 : NumPy ndarray
    -   1D array of low frequencies most recently used with `compute`.

    f2 : NumPy ndarray
    -   1D array of high frequencies most recently used with `compute`.

    verbose : bool; default True
    -   Whether or not to report the progress of the processing.
    """

    def compute(
        self,
        indices: tuple[np.ndarray] | None = None,
        f1: np.ndarray | None = None,
        f2: np.ndarray | None = None,
        n_jobs: int = 1,
    ) -> None:
        """Compute PPC, averaged over epochs.

        PARAMETERS
        ----------
        indices: tuple of NumPy ndarray of int | None; default None
        -   Indices of the channels to compute PPC between. Should contain 2
            1D arrays of equal length for the seed and target indices,
            respectively. If None, coupling between all channels is computed.

        f1 : numpy ndarray | None; default None
        -   A 1D array of the lower frequencies to compute PPC on. If None, all
            frequencies are used.

        f2 : numpy ndarray | None; default None
        -   A 1D array of the higher frequencies to compute PPC on. If None,
            all frequencies are used.

        n_jobs : int; default 1
        -   The number of jobs to run in parallel.

        NOTES
        -----
        -   PPC is computed between all values of `f1` and `f2`. If any value
            of `f1` is higher than `f2`, a NaN value is returned.
        """
        self._sort_indices(indices)
        self._sort_freqs(f1, f2)

        if self.verbose:
            print("Computing PPC on the data...")

        self._compute_ppc_multi(n_jobs)

        if self.verbose:
            print("    [PPC computation finished]\n")

    def _compute_ppc_multi(self, n_jobs: int) -> None:
        """Compute PPC between f1s of seeds and f2s of targets."""
        args = [
            {
                "data": self.data[
                    :, (self._seeds[con_i], self._targets[con_i])
                ],
                "freqs": self.freqs,
                "f1s": self.f1,
                "f2s": self.f2,
                "results": np.full(
                    (self.f2.shape[0], self.f1.shape[0]),
                    fill_value=np.nan,
                ),
            }
            for con_i in range(self._n_cons)
        ]

        self._results = np.array(
            pqdm(
                args,
                _compute_ppc,
                n_jobs,
                argument_type="kwargs",
                desc="Processing connections...",
                disable=not self.verbose,
            )
        )

    def get_results(
        self, form: str = "raveled"
    ) -> np.ndarray | tuple[np.ndarray, tuple[np.ndarray]]:
        """Return a copy of the results.

        PARAMETERS
        ----------
        form : str; default "raveled"
        -   How the results should be returned: "raveled" - results have shape
            [connections x f2 x f1]; "compact" - results have shape [seeds x
            targets x f2 x f1].

        RETURNS
        -------
        results : NumPy ndarray
        -   Spectral coupling results.

        indices : tuple of NumPy ndarray
        -   Channel indices of the seeds and targets. Only returned if `form`
            is "compact".
        """
        accepted_forms = ["raveled", "compact"]
        if form not in accepted_forms:
            raise ValueError("`form` is not recognised.")

        if form == "raveled":
            return self._results.copy()
        else:
            return self._get_compact_results(self._results)


@njit
def _compute_ppc(
    data: np.ndarray,
    freqs: np.ndarray,
    f1s: np.ndarray,
    f2s: np.ndarray,
    results: np.ndarray,
) -> np.ndarray:
    """Compute PPC for a single connection across epochs.

    PARAMETERS
    ----------
    data : NumPy ndarray
    -   FFT coefficients with shape [epochs x 2 x frequencies], where the
        second dimension contains the data for the seed and target channel of
        a single connection, respectively.

    freqs : NumPy ndarray
    -   Frequencies of `data`.

    f1s : NumPy ndarray
    -   Low frequency values in Hz.

    f2s : NumPy ndarray
    -   High frequency values in Hz.

    results : NumPy ndarray
    -   Array of shape [f2 x f1] to store the results in.
    """
    for f1_i, f1 in enumerate(f1s):
        for f2_i, f2 in enumerate(f2s):
            if f1 < f2:
                fft_f1 = data[:, 0, np.where(freqs == f1)[0][0]]  # seed f1
                fft_f2 = data[:, 1, np.where(freqs == f2)[0][0]]  # target f2
                numerator = np.abs(
                    (
                        np.abs(fft_f1)
                        * np.abs(fft_f2)
                        * np.exp(
                            1j
                            * (
                                np.angle(fft_f1, True) * (f2 / f1)
                                - np.angle(fft_f2, True)
                            )
                        )
                    ).mean()
                )
                denominator = (np.abs(fft_f1) * np.abs(fft_f2)).mean()
                results[f2_i, f1_i] = numerator / denominator

    return results
