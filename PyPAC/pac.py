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
    """

    def __init__(self, data: np.ndarray, sfreq: float) -> None:
        self._data = data.copy()
        self._sfreq = copy.copy(sfreq)

        self._sort_init_inputs()
        self._init_attrs()

    def _sort_init_inputs(self) -> None:
        """Check init. inputs are appropriate."""
        if not isinstance(self._data, np.ndarray) or self._data.ndim != 3:
            raise ValueError("`data` must be a 3D NumPy array.")

        if self._sfreq <= 0:
            raise ValueError("`sfreq` must be > 0.")

    def _init_attrs(self) -> None:
        """Initialise attrs. from the data."""
        self._n_epochs, _, self._n_times = self._data.shape
        self._freqs = np.linspace(0, self._sfreq / 2, self._sfreq + 1).tolist()
        self._n_freqs = len(self._freqs)

    def compute_pac(
        self,
        low_freqs: list[float],
        high_freqs: list[float],
        seeds: list[int],
        targets: list[int],
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
        self._normalise = copy.copy(normalise)
        self._sort_freq_inputs(low_freqs, high_freqs)
        self._sort_indices_inputs(seeds, targets)

        self._compute_fft()
        self._compute_bispectra()
        if self._normalise:
            self._normalise_bispectra()

    def _sort_freq_inputs(
        self, low_freqs: list[float], high_freqs: list[float]
    ) -> None:
        """Sort low and high freq. inputs."""
        self._n_freq_pairs = len(low_freqs) * len(high_freqs)

        low_freq_idcs = np.zeros((2, self._n_freq_pairs), dtype=int)
        high_freq_idcs = np.zeros((2, self._n_freq_pairs), dtype=int)

        pair_i = 0
        for lfreq in low_freqs:
            for hfreq in high_freqs:
                low_freq_idcs[0, pair_i] = self._freqs.index(lfreq)
                low_freq_idcs[1, pair_i] = self._freqs.index(hfreq - lfreq)
                high_freq_idcs[0, pair_i] = self._freqs.index(lfreq)
                high_freq_idcs[1, pair_i] = self._freqs.index(hfreq)
                pair_i += 1

        self._low_freq_idcs = low_freq_idcs
        self._high_freq_idcs = high_freq_idcs

    def _sort_indices_inputs(
        self, seeds: list[int], targets: list[int]
    ) -> None:
        """Sort seed and target inputs."""
        seeds = copy.deepcopy(seeds)
        targets = copy.deepcopy(targets)

        if seeds is None:
            seeds = np.arange(self._n_chs, dtype=int).tolist()
        if targets is None:
            targets = np.arange(self._n_chs, dtype=int).tolist()

        if any(0 > seed >= self._n_chans for seed in seeds) or any(
            0 > target >= self._n_chans for target in targets
        ):
            raise ValueError(
                "`seeds` and/or `targets` contain channels not in the data."
            )

        self._use_chans = np.unique([*seeds, *targets]).tolist()
        self._n_chans = len(self._use_chans)
        self._n_chan_pairs = len(seeds) * len(targets)

        self._seeds = np.searchsorted(self._use_chans, seeds).tolist()
        self._targets = np.searchsorted(self._use_chans, targets).tolist()

        self._original_seeds = seeds
        self._original_targets = targets

    def _compute_fft(self) -> None:
        """Compute FFT of the data."""
        self._fft_data = np.linalg.fft(
            sp.signal.detrend(self._data[self._use_chans])
            * np.hanning(self._n_times)
        )

    def _compute_bispectra(self) -> None:
        """Compute the bispectra for two channels."""
        bispectra = np.zeros((2, 2, 2, self._n_freq_pairs, 2))

        fft_coeffs_lfreqs = self._get_fft_coeffs(self._low_freq_idcs)
        fft_coeffs_hfreqs = self._get_fft_coeffs(self._high_freq_idcs)

        bispectrum_lfreqs = self._compute_bispectrum(fft_coeffs_lfreqs)
        bispectrum_hfreqs = self._compute_bispectrum(fft_coeffs_lfreqs)

    def _get_fft_coeffs(
        self, freqs: np.ndarray
    ) -> np.ndarray:
        """Get FFT coefficients for two channels.

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
        -   Array of FFT coefficients with shape [4 x epochs x chan. pairs x
            freq. pairs]. The first dimension corresponds to (for each freq.
            pair) the low freq., high freq. - low freq., high freq., and high
            freq. + low freq., respectively.
        """
        fft_coeffs = np.zeros((4, self._n_epochs, self._n_chan_pairs, self._n_freq_pairs))

        freq_pair_i = 0
        for lfreq in freqs[0]:
            for hfreq in freqs[1]:
                fft_coeffs[0, :, :, freq_pair_i] = self._fft_data[..., lfreq]
                fft_coeffs[1, :, :, freq_pair_i] = self._fft_data[..., hfreq - lfreq]
                fft_coeffs[2, :, :, freq_pair_i] = self._fft_data[..., hfreq]
                fft_coeffs[0, :, :, freq_pair_i] = self._fft_data[..., hfreq + lfreq]

                freq_pair_i += 1

        return fft_coeffs

    def _compute_bispectrum(self, fft_coeffs: np.ndarray) -> np.ndarray:
        """"""
        bispectrum = np.zeros((2, 2, 2, self._n_freq_pairs, 2))

        bispectrum[..., 0] = fft_coeffs[]

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
