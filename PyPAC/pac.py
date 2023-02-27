"""Class for computing PAC from the bispectra."""

import copy
import numpy as np
import scipy as sp


class BispectraPAC:
    """Compute phase amplitude coupling (PAC) using the bispectra.

    PARAMETERS
    ----------
    data : numpy ndarray
    -   Array of data to compute PAC on, with shape [epochs x channels x
        times].

    sfreq : float
    -   Sampling frequency of the data in Hz.

    freq_res : float | None; default None
    -   Desired frequency resolution of the results in Hz. If None, the
        resolution is based on `sfreq`.
    """

    def __init__(
        self, data: np.ndarray, sfreq: float, freq_res: float | None = None
    ) -> None:
        self._data = data.copy()
        self._sfreq = copy.copy(sfreq)
        self._freq_res = copy.copy(freq_res)

        self._sort_init_inputs()
        self._init_attrs()

    def _sort_init_inputs(self) -> None:
        """Check init. inputs are appropriate."""
        if not isinstance(self._data, np.ndarray) or self._data.ndim != 3:
            raise ValueError("`data` must be a 3D NumPy array.")

        if self._sfreq <= 0:
            raise ValueError("`sfreq` must be > 0.")

        if self._freq_res is not None and self._freq_res <= 0:
            raise ValueError("`freq_res` must be > 0.")

    def _init_attrs(self) -> None:
        """Initialise attrs. from the data."""
        if self._freq_res is None:
            self._freqs = np.linspace(
                0, self._sfreq / 2, self._sfreq + 1
            ).tolist()
        else:
            self._freqs = np.arange(
                0, self._sfreq / 2 + self._freq_res, self._freq_res
            ).tolist()

        self._n_freqs = len(self._freqs)

        self._n_epochs, self._n_chans, self._n_times = self._data.shape

    def compute_pac(
        self,
        low_freqs: list[float],
        high_freqs: list[float],
        seeds: list[int] | None = None,
        targets: list[int] | None = None,
        normalise: bool = True,
    ) -> None:
        """Compute PAC on the data.

        PARAMETERS
        ----------
        low_freqs : list of float
        -   The low frequencies for PAC.

        high_freqs : list of float
        -   The high frequencies for PAC.

        seeds : list of int | None; default None
        -   Indices of channels in the data to treat as seeds.

        targets : list of int | None; default None
        -   Indices of channels in the data to treat as targets.

        normalise : bool; default True
        -   Whether or not to normalise the bispectrum to bicoherence using the
            threenorm.
        """
        self._sort_freq_inputs(low_freqs, high_freqs)
        self._sort_indices_inputs(seeds, targets)

        self._compute_fft()
        self._compute_bispectra()
        if normalise:
            self._normalise_bispectra()
        self._compute_pac()

    def _sort_freq_inputs(
        self, low_freqs: list[float], high_freqs: list[float]
    ) -> None:
        """Sort low and high freq. inputs."""
        if len(low_freqs) != len(high_freqs):
            raise ValueError(
                "`low_freqs` and `high_freqs` must have equal length."
            )
        self._n_freq_pairs = len(low_freqs)

        low_freq_idcs = np.zeros((2, self._n_freq_pairs), dtype=int)
        high_freq_idcs = np.zeros((2, self._n_freq_pairs), dtype=int)

        freq_i = 0
        for lfreq, hfreq in zip(low_freqs, high_freqs):
            low_freq_idcs[0, freq_i] = self._freqs.index(lfreq)
            low_freq_idcs[1, freq_i] = self._freqs.index(hfreq - lfreq)
            high_freq_idcs[0, freq_i] = self._freqs.index(lfreq)
            high_freq_idcs[1, freq_i] = self._freqs.index(hfreq)

            freq_i += 1

        self._low_freq_idcs = low_freq_idcs
        self._high_freq_idcs = high_freq_idcs

    def _sort_indices_inputs(
        self, seeds: list[int], targets: list[int]
    ) -> None:
        """Sort seed and target inputs."""
        seeds = copy.deepcopy(seeds)
        targets = copy.deepcopy(targets)

        if seeds is None:
            seeds = np.arange(self._n_chans, dtype=int).tolist()
        if targets is None:
            targets = np.arange(self._n_chans, dtype=int).tolist()

        if any(0 > seed >= self._n_chans for seed in seeds) or any(
            0 > target >= self._n_chans for target in targets
        ):
            raise ValueError(
                "`seeds` and/or `targets` contain channels not in the data."
            )

        self._use_chans = np.unique([*seeds, *targets]).tolist()
        self._n_chans = len(self._use_chans)

        self._seeds = np.searchsorted(self._use_chans, seeds).tolist()
        self._targets = np.searchsorted(self._use_chans, targets).tolist()
        self._n_seeds = len(self._seeds)
        self._n_targets = len(self._targets)

        self._original_seeds = seeds
        self._original_targets = targets

    def _compute_fft(self) -> None:
        """Compute FFT of the data."""
        self._fft_data = np.fft.fft(
            sp.signal.detrend(self._data[:, self._use_chans])
            * np.hanning(self._n_times)
        )

    def _compute_bispectra(self) -> None:
        """Compute the bispectra for multiple seeds and targets."""
        fft_coeffs_lfreqs = self._get_fft_coeffs(self._low_freq_idcs)
        fft_coeffs_hfreqs = self._get_fft_coeffs(self._high_freq_idcs)

        self._bispectra_lfreqs = np.zeros(
            (self._n_seeds, self._n_targets, 2, 2, 2, self._n_freq_pairs, 2),
            dtype=np.complex128,
        )
        self._bispectra_hfreqs = np.zeros(
            (self._n_seeds, self._n_targets, 2, 2, 2, self._n_freq_pairs, 2),
            dtype=np.complex128,
        )
        for seed_i, seed in enumerate(self._seeds):
            for target_i, target in enumerate(self._targets):
                self._bispectra_lfreqs[
                    seed_i, target_i
                ] = self._compute_seed_target_bispectra(
                    fft_coeffs_lfreqs, seed, target
                )
                self._bispectra_hfreqs[
                    seed_i, target_i
                ] = self._compute_seed_target_bispectra(
                    fft_coeffs_hfreqs, seed, target
                )

    def _get_fft_coeffs(self, freqs: np.ndarray) -> np.ndarray:
        """Get FFT coefficients for multiple channels.

        PARAMETERS
        ----------
        freqs : numpy ndarray
        -   Array of frequencies to get the FFT coefficients for with shape [2
            x N], where the first and second rows correspond to a set of lower
            and higher frequencies, respectively, and N corresponds to the
            number of frequency pairs.

        RETURNS
        -------
        fft_coeffs : numpy ndarray
        -   Array of FFT coefficients with shape [4 x epochs x channels x freq.
            pairs]. The first dimension corresponds to (for each freq. pair)
            the low freq., high freq. - low freq., high freq., and high freq. +
            low freq., respectively.
        """
        fft_coeffs = np.zeros(
            (4, self._n_epochs, self._n_chans, self._n_freq_pairs),
            dtype=np.complex128,
        )

        freq_i = 0
        for lfreq, hfreq in zip(freqs[0], freqs[1]):
            fft_coeffs[0, :, :, freq_i] = self._fft_data[..., lfreq]
            fft_coeffs[1, :, :, freq_i] = self._fft_data[..., hfreq - lfreq]
            fft_coeffs[2, :, :, freq_i] = self._fft_data[..., hfreq]
            fft_coeffs[3, :, :, freq_i] = self._fft_data[..., hfreq + lfreq]

            freq_i += 1

        return fft_coeffs

    def _compute_seed_target_bispectra(
        self, fft_coeffs: np.ndarray, seed: int, target: int
    ) -> np.ndarray:
        """Compute bispectra for single seed and target channels.

        PARAMETERS
        ----------
        ...
        seed : int
        -   Index of the seed channel.

        target : int
        -   Index of the target channel.

        RETURNS
        -------
        bispectra : numpy ndarray
        -   Bispectra array with shape [channels x channels x channels x freq.
            pairs x 2], where the channels are always 2 (i.e. one seed and one
            target) and the final dimension corresponds to the 'f1, f2-f1, f2'
            and 'f1, f2, f1+f2' bispectra, respectively.
        """
        bispectra = np.zeros(
            (self._n_epochs, 2, 2, 2, self._n_freq_pairs, 2),
            dtype=np.complex128,
        )

        for freq_i in range(self._n_freq_pairs):
            for epoch_i in range(self._n_epochs):
                epoch_coeffs = fft_coeffs[:, epoch_i, (seed, target), freq_i]
                for chan_i, chan in enumerate((seed, target)):
                    bispectra[epoch_i, :, :, chan_i, freq_i, 0] = (
                        epoch_coeffs[0][np.newaxis, :].T
                        @ epoch_coeffs[1][np.newaxis, :]
                        * epoch_coeffs[2, chan].conj()
                    )
                    bispectra[epoch_i, :, :, chan_i, freq_i, 1] = (
                        epoch_coeffs[0][np.newaxis, :].T
                        @ epoch_coeffs[2][np.newaxis, :]
                        * epoch_coeffs[3, chan].conj()
                    )

        return bispectra.mean(axis=0)

    @property
    def indices(self) -> tuple[list[int]]:
        """Return indices of the seeds and targets."""
        return (self._original_seeds, self._original_targets)

    @property
    def results(self) -> np.ndarray:
        """Return PAC results."""
        if self._normalise:
            return self._bicoherence
        return self._bispectra
