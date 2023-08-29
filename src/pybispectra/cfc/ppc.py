"""Tools for handling PPC analysis."""

from copy import deepcopy

import numpy as np
from numba import njit
from pqdm.processes import pqdm

from pybispectra.utils import ResultsCFC
from pybispectra.utils._process import _ProcessFreqBase
from pybispectra.utils._utils import _fast_find_first


class PPC(_ProcessFreqBase):
    """Class for computing phase-phase coupling (PPC).

    Parameters
    ----------
    data : numpy.ndarray of float, shape of [epochs, channels, frequencies]
        Fourier coefficients.

    freqs : numpy.ndarray of float, shape of [frequencies]
        Frequencies (in Hz) in :attr:`data`.

    verbose : bool (default True)
        Whether or not to report the progress of the processing.

    Methods
    -------
    compute
        Compute PPC, averaged over epochs.

    Attributes
    ----------
    results : tuple of pybispectra.ResultsCFC
        PPC results for each of the computed metrics.

    data : numpy.ndarray of float, shape of [epochs, channels, frequencies]
        FFT coefficients.

    freqs : numpy.ndarray of float, shape of [frequencies]
        Frequencies (in Hz) in :attr:`data`.

    indices : tuple of tuple of int, length of 2
        Indices of the seed and target channels, respectively, most recently
        used with :meth:`compute`.

    f1s : numpy.ndarray of float, shape of [frequencies]
        Low frequencies (in Hz) most recently used with :meth:`compute`.

    f2s : numpy.ndarray of float, shape of [frequencies]
        High frequencies (in Hz) most recently used with :meth:`compute`.

    verbose : bool
        Whether or not to report the progress of the processing.
    """

    _ppc = None

    def compute(
        self,
        indices: tuple[tuple[int], tuple[int]] | None = None,
        f1s: np.ndarray | None = None,
        f2s: np.ndarray | None = None,
        n_jobs: int = 1,
    ) -> None:
        r"""Compute PPC, averaged over epochs.

        Parameters
        ----------
        indices : tuple of tuple of int, length of 2 | None (default None)
            Indices of the seed and target channels, respectively, to compute
            PPC between. If ``None``, coupling between all channels is
            computed.

        f1s : numpy.ndarray | None (default None), shape of [frequencies]
            Lower frequencies to compute PPC on. If ``None``, all frequencies
            are used.

        f2s : numpy.ndarray | None (default None), shape of [frequencies]
            Higher frequencies to compute PPC on. If ``None``, all frequencies
            are used.

        n_jobs : int (default ``1``)
            Number of jobs to run in parallel. If ``-1``, all available CPUs
            are used.

        Notes
        -----
        PPC is computed as coherence between frequencies :footcite:`Giehl2021`:

        :math:`\large PPC(\vec{x}_{f_1},\vec{y}_{f_2})=\Large \frac{|\langle
        \vec{a}_x(f_1)\vec{a}_y(f_2) e^{i(\vec{\varphi}_x(f_1)\frac{f_2}{f_1}-
        \vec{\varphi}_y(f_2))} \rangle|}{\langle \vec{a}_x(f_1)\vec{a}_y(f_2)
        \rangle}`,

        where :math:`\vec{a}(f)` and :math:`\vec{\varphi}(f)` are the amplitude
        and phase of a signal at a given frequency, respectively, and the
        angled brackets represent the average over epochs.

        PPC is computed between all values of :attr:`f1s` and :attr:`f2s`. If
        any value of :attr:`f1s` is higher than :attr:`f2s`, a ``numpy.nan``
        value is returned.

        References
        ----------
        .. footbibliography::
        """
        self._reset_attrs()

        self._sort_indices(indices)
        self._sort_freqs(f1s, f2s)
        self._sort_parallelisation(n_jobs)

        if self.verbose:
            print("Computing PPC...")

        self._compute_ppc()
        self._store_results()

        if self.verbose:
            print("    ... PPC computation finished\n")

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
                "f1s": self.f1s,
                "f2s": self.f2s,
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
            self._ppc, self.indices, self.f1s, self.f2s, "PPC"
        )

    @property
    def results(self) -> tuple[ResultsCFC]:
        """Return the results."""
        return deepcopy(self._results)


@njit
def _compute_ppc(
    data: np.ndarray,
    freqs: np.ndarray,
    f1s: np.ndarray,
    f2s: np.ndarray,
) -> np.ndarray:
    """Compute PPC for a single connection across epochs.

    Parameters
    ----------
    data : numpy.ndarray, shape of [epochs, 2, frequencies]
        FFT coefficients where the second dimension contains the data for the
        seed and target channel of a single connection, respectively.

    freqs : numpy.ndarray, shape of [frequencies]
        Frequencies in ``data``.

    f1s : numpy.ndarray, shape of [frequencies]
        Low frequencies to compute coupling for.

    f2s : numpy.ndarray, shape of [frequencies]
        High frequencies to compute coupling for.

    Returns
    -------
    results : numpy.ndarray, shape of [f1s, f2s]
        PPC for a single connection.
    """
    results = np.full(
        (f1s.shape[0], f2s.shape[0]), fill_value=np.nan, dtype=np.float64
    )
    f1_idx = 0  # starting index to find f1s
    for f1_i, f1 in enumerate(f1s):
        f2_idx = 0  # starting index to find f2s
        for f2_i, f2 in enumerate(f2s):
            if f1 < f2 and f1 > 0:
                fft_f1 = data[:, 0, _fast_find_first(freqs, f1, f1_idx)]
                fft_f2 = data[:, 1, _fast_find_first(freqs, f2, f2_idx)]
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
