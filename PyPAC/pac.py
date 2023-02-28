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

    _pac_bispec_standard = None
    _pac_bispec_antisym = None
    _pac_bicoh_standard = None
    _pac_bicoh_antisym = None

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
        compute_antisym: bool = True,
        compute_bicoh: bool = True,
    ) -> np.ndarray | tuple[np.ndarray]:
        """Compute PAC from the bispectra and/or bicoherence.

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

        compute_antisym : bool; default True
        -   Whether or not to compute the antisymmetrised results.

        compute_bicoh : bool; default True
        -   Whether or not to compute the bicoherence (bispectrum normalised
            with the threenorm).

        RETURNS
        -------
        results : numpy ndarray | tuple of numpy ndarray
        -   The computed PAC results derived from the: standard bispectra;
            antisymmetrised bispectra (if `compute_antisym` = True); standard
            bicoherence (if `compute_bicoh` = True); and antisymmetrised
            bicoherence (if `compute_antisym` and `compute_bicoh` = True),
            respectively.
        """
        self._reset_results()

        self._bicoh = copy.copy(compute_bicoh)
        self._antisym = copy.copy(compute_antisym)
        self._sort_freq_inputs(low_freqs, high_freqs)
        self._sort_indices_inputs(seeds, targets)

        self._compute_fft()
        self._compute_bispectra()
        self._compute_bispectra_pac()
        if self._bicoh:
            self._compute_bicoherence()
            self._compute_bicoherence_pac()

        return self.results

    def _reset_results(self) -> None:
        """Reset result attrs. to None."""
        self._pac_bispec_standard = None
        self._pac_bispec_antisym = None
        self._pac_bicoh_standard = None
        self._pac_bicoh_antisym = None

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
        self._bispectra_lfreqs = self._compute_bispectra_from_fft(
            self._get_fft_coeffs(self._low_freq_idcs)
        )
        self._bispectra_hfreqs = self._compute_bispectra_from_fft(
            self._get_fft_coeffs(self._high_freq_idcs)
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

    def _compute_bispectra_from_fft(
        self, fft_coeffs: np.ndarray
    ) -> np.ndarray:
        """Compute bispectra from FFT coeffs. averaged across epochs.

        PARAMETERS
        ----------
        fft_coeffs : numpy ndarray

        RETURNS
        -------
        bispectra : numpy ndarray
        -   Bispectra array with shape [channels x channels x channels x freq.
            pairs x 2], where the channels are the seed and target channels and
            the final dimension corresponds to the 'f1, f2-f1, f2' and
            'f1, f2, f1+f2' bispectra, respectively.
        """
        bispectra = np.zeros(
            (
                self._n_epochs,
                self._n_chans,
                self._n_chans,
                self._n_chans,
                self._n_freq_pairs,
                2,
            ),
            dtype=np.complex128,
        )

        for freq_i in range(self._n_freq_pairs):
            for epoch_i in range(self._n_epochs):
                epoch_coeffs = fft_coeffs[:, epoch_i, :, freq_i]
                for chan_i, chan in enumerate(self._use_chans):
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

    def _compute_bispectra_pac(self) -> None:
        """"""
        pac_standard = np.zeros(
            (self._n_seeds, self._n_targets, self._n_freq_pairs)
        )
        for seed in self._seeds:
            for target in self._targets:
                standard_lfreqs = np.abs(
                    [
                        self._bispectra_lfreqs[seed, target, target, :, 0],
                        self._bispectra_lfreqs[target, seed, seed, :, 0],
                    ]
                )
                standard_hfreqs = np.abs(
                    [
                        self._bispectra_hfreqs[seed, target, target, :, 0],
                        self._bispectra_hfreqs[target, seed, seed, :, 0],
                    ]
                )

                pac_standard[seed, target] = np.mean(
                    [standard_lfreqs[0], standard_hfreqs[0]]
                )
                pac_standard[target, seed] = np.mean(
                    [standard_lfreqs[1], standard_hfreqs[1]]
                )

    @property
    def indices(self) -> tuple[list[int]]:
        """Return indices of the seeds and targets."""
        return (self._original_seeds, self._original_targets)

    @property
    def results(
        self, get_types: str | list[str] | None = None
    ) -> np.ndarray | None | tuple[np.ndarray | None]:
        """Return PAC results.

        PARAMETERS
        ----------
        get_types : str | list of str | None; default None
        -   The type of PAC results to return. Recognised values are:
            "bispec_standard" for the standard bispectra results;
            "bispec_antisym" for the antisymmetrised bispectra results;
            "bicoh_standard" for the standard bicoherence results; and
            "bicoh_antisym" for the antisymmetrised bicoherence results. If
            None, all computed results are returned.

        RETURNS
        -------
        results : numpy ndarray | None | tuple of numpy ndarray or None
        -   The requested results. If `get_types` is None, only those results
            which have been computedare returned, whereas if a result is
            requested which has not been computed, None is returned in its
            place.

        RAISES
        ------
        KeyError
        -   Raised if key(s) of `get_types` is(are) invalid.
        """
        results = {
            "bispec_standard": self._pac_bispec_standard,
            "bispec_antisym": self._pac_bispec_antisym,
            "bicoh_standard": self._pac_bicoh_standard,
            "bicoh_antisym": self._pac_bicoh_antisym,
        }
        possible_types = results.keys()

        if get_types is None:
            return (
                results[this_type]
                for this_type in possible_types
                if results[this_type] is not None
            )

        if isinstance(get_types, str):
            get_types = [get_types]

        try:
            return (results[this_type] for this_type in get_types)
        except KeyError as error:
            print("The requested results type is not recognised.", error)

    @property
    def pac_bispec_standard(self) -> np.ndarray | None:
        """Return PAC results from the standard bispectra."""
        return self._pac_bispec_standard

    @property
    def pac_bispec_antisym(self) -> np.ndarray | None:
        """Return PAC results from the antisymmetrised bispectra."""
        return self._pac_bispec_antisym

    @property
    def pac_bicoh_standard(self) -> np.ndarray | None:
        """Return PAC results from the standard bicoherence."""
        return self._pac_bicoh_standard

    @property
    def pac_bicoh_antisym(self) -> np.ndarray | None:
        """Return PAC results from the antisymmetrised bispectra."""
        return self._pac_bicoh_antisym
