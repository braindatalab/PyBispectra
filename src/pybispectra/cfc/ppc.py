"""Tools for handling PPC analysis."""

import numpy as np
from numba import njit
from pqdm.processes import pqdm

from pybispectra.utils import ResultsCFC, fast_find_first
from pybispectra.utils._process import _ProcessFreqBase


class PPC(_ProcessFreqBase):
    """Class for computing phase-phase coupling (PPC).

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
        PPC results for each of the computed metrics.

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

    _ppc = None

    def compute(
        self,
        indices: tuple[np.ndarray] | None = None,
        f1: np.ndarray | None = None,
        f2: np.ndarray | None = None,
        n_jobs: int = 1,
    ) -> None:
        r"""Compute PPC, averaged over epochs.

        Parameters
        ----------
        indices : tuple of numpy.ndarray of int | None (default None)
            Indices of the channels to compute PPC between. Should contain two
            1D arrays of equal length for the seed and target indices,
            respectively. If ``None``, coupling between all channels is
            computed.

        f1 : numpy.ndarray of float | None (default None)
            1D array of the lower frequencies to compute PPC on. If ``None``,
            all frequencies are used.

        f2 : numpy.ndarray of float | None; default None
            1D array of the higher frequencies to compute PPC on. If ``None``,
            all frequencies are used.

        n_jobs : int (default ``1``)
            Number of jobs to run in parallel.

        Notes
        -----
        PPC is computed as coherence between frequencies :footcite:`Giehl2021`:

        :math:`\large PPC(\vec{x}_{f_1},\vec{y}_{f_2})=\Large \frac{|\langle \vec{a}_x(f_1)\vec{a}_y(f_2) e^{i(\vec{\varphi}_x(f_1)\frac{f_2}{f_1}-\vec{\varphi}_y(f_2))} \rangle|}{\langle \vec{a}_x(f_1)\vec{a}_y(f_2) \rangle}`,

        where :math:`\vec{a}(f)` and :math:`\vec{\varphi}(f)` are the amplitude and phase
        of a signal at a given frequency, respectively, and the angled brackets
        represent the average over epochs.

        PPC is computed between all values of :attr:`f1` and :attr:`f2`. If any
        value of :attr:`f1` is higher than :attr:`f2`, a ``numpy.nan`` value is
        returned.

        References
        ----------
        .. footbibliography::
        """  # noqa E501
        self._reset_attrs()

        self._sort_indices(indices)
        self._sort_freqs(f1, f2)
        self._sort_parallelisation(n_jobs)

        if self.verbose:
            print("Computing PPC...")

        self._compute_ppc()
        self._store_results()

        if self.verbose:
            print("    [PPC computation finished]\n")

    def _reset_attrs(self) -> None:
        """Reset attrs. of the object to prevent interference."""
        super()._reset_attrs()
        self._ppc = None

    def _compute_ppc(self) -> None:
        """Compute PPC between f1s of seeds and f2s of targets."""
        args = [
            {
                "data": self.data[:, (seed, target)],
                "freqs": self.freqs,
                "f1s": self.f1,
                "f2s": self.f2,
            }
            for seed, target in zip(self._seeds, self._targets)
        ]

        self._ppc = np.array(
            pqdm(
                args,
                _compute_ppc,
                self._n_jobs,
                argument_type="kwargs",
                desc="Processing connections...",
                disable=not self.verbose,
            )
        )

    def _store_results(self) -> None:
        """Store computed results in an object."""
        self._results = ResultsCFC(
            self._ppc, self.indices, self.f1, self.f2, "PPC"
        )

    @property
    def results(self) -> tuple[ResultsCFC]:
        """Return the results."""
        return self._results


@njit
def _compute_ppc(
    data: np.ndarray,
    freqs: np.ndarray,
    f1s: np.ndarray,
    f2s: np.ndarray,
) -> np.ndarray:
    """Compute PPC for a single connection across epochs.

    PARAMETERS
    ----------
    data : numpy.ndarray of float
        3D array of FFT coefficients with shape `[epochs x 2 x frequencies]`,
        where the second dimension contains the data for the seed and target
        channel of a single connection, respectively.

    freqs : numpy.ndarray of float
        1D array of frequencies in ``data``.

    f1s : numpy.ndarray of float
        1D array of low frequencies to compute coupling for.

    f2s : numpy.ndarray of float
        1D array of high frequencies to compute coupling for.

    RETURNS
    -------
    results : numpy.ndarray of float
        2D array of PPC for a single connection with shape `[f1 x f2]`.
    """
    results = np.full(
        (f1s.shape[0], f2s.shape[0]), fill_value=np.nan, dtype=np.float64
    )
    for f1_i, f1 in enumerate(f1s):
        for f2_i, f2 in enumerate(f2s):
            if f1 < f2 and f1 > 0:
                fft_f1 = data[:, 0, fast_find_first(freqs, f1)]  # seed f1
                fft_f2 = data[:, 1, fast_find_first(freqs, f2)]  # target f2
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
                results[f1_i, f2_i] = numerator / denominator

    return results
