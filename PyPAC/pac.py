"""Class for computing PAC from the bispectra."""

import copy

import numpy as np
import scipy as sp

from classes import Process, Result


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

    _pac_bispec_standard = Result
    _pac_bispec_antisym = Result
    _pac_bicoh_standard = Result
    _pac_bicoh_antisym = Result

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
        low_freqs: list[list[float]],
        high_freqs: list[list[float]],
        seeds: list[int] | None = None,
        targets: list[int] | None = None,
        symmetrisation: list[str] = ["none", "antisym"],
        normalisation: list[str] = ["none", "threenorm"]
    ) -> np.ndarray | tuple[np.ndarray]:
        """Compute PAC from the bispectra and/or bicoherence.

        PARAMETERS
        ----------
        low_freqs : list of list of float
        -   The low frequencies of each band for PAC.

        high_freqs : list of list of float
        -   The high frequencies of each band for PAC.

        seeds : list of int | None; default None
        -   Indices of channels in the data to treat as seeds.

        targets : list of int | None; default None
        -   Indices of channels in the data to treat as targets.

        symmetrisation : list[str]; default ["none", "antisymmetrised"]
        -   The symmetrisation to perform when computing PAC.

        normalisation : list[str]; default ["none", "threenorm"]
        -   The normalisation to perform when computing PAC.

        RETURNS
        -------
        results : Results | list[Results]
        -   Object or list of objects containing the PAC results.
        """
        self._reset_results()

        self._bicoh = copy.copy(compute_bicoh)
        self._antisym = copy.copy(compute_antisym)
        self._low_freqs = copy.deepcopy(low_freqs)
        self._high_freqs = copy.deepcopy(high_freqs)

        self._sort_freq_inputs()
        self._sort_indices_inputs(seeds, targets)

        self._sort_results_containers()

        self._compute_fft()

        self._compute_bispectra()
        self._compute_bivariance()
        if self._bicoh:
            self._compute_bicoherence()
        
        self._pac_bivar_standard = 

        return self.results

    def _reset_results(self) -> None:
        """Reset result attrs. to None."""
        self._pac_bispec_standard = 
        self._pac_bispec_antisym = None
        self._pac_bicoh_standard = None
        self._pac_bicoh_antisym = None

    def _sort_freq_inputs(self) -> None:
        """Sort low and high freq. inputs."""
        if len(self._low_freqs) != len(self._high_freqs):
            raise ValueError(
                "`low_freqs` and `high_freqs` must have equal length."
            )
        self._n_bands = len(self._low_freqs)

        freq_inds_down = []
        freq_inds_up = []
        for low_band, high_band in zip(self._low_freqs, self._high_freqs):
            down_idcs = [[], []]
            up_idcs = [[], []]
            for lfreq in low_band:
                for hfreq in high_band:
                    down_idcs[0].append(self._freqs.index(lfreq))
                    down_idcs[1].append(self._freqs.index(hfreq - lfreq))
                    up_idcs[0].append(self._freqs.index(lfreq))
                    up_idcs[1].append(self._freqs.index(hfreq))
            freq_inds_down.append(down_idcs)
            freq_inds_up.append(up_idcs)

        self._freq_inds_down = freq_inds_down
        self._freq_inds_up = freq_inds_up

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

        self._original_seeds = seeds
        self._original_targets = targets

    def _sort_results_containers(self) -> None:
        """"""
        n_result_types = 1
        if self._antisym or self._bicoh:
            n_result_types *= 2
            if self._antisym and self._bicoh:
                n_result_types *= 2

        results = []
        for low_band, high_band in zip(self._low_freqs, self._high_freqs):
            results.append(
                [
                    np.zeros(
                        (
                            len(self._seeds),
                            len(self._targets),
                            len(low_band),
                            len(high_band),
                        )
                    )
                    for _ in range(n_result_types)
                ]
            )

        self._results = results

    def _compute_fft(self) -> None:
        """Compute FFT of the data."""
        self._fft_data = np.fft.fft(
            sp.signal.detrend(self._data[:, self._use_chans])
            * np.hanning(self._n_times)
        )

    def _compute_bispectra(self) -> None:
        """Compute the bispectra for multiple seeds and targets."""
        self._bispectra_down = self._compute_bispectra_from_fft(
            self._get_fft_coeffs(self._freq_inds_down)
        )
        self._bispectra_up = self._compute_bispectra_from_fft(
            self._get_fft_coeffs(self._freq_inds_up)
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
        fft_coeffs = []
        for fband in freqs:
            coeffs = np.zeros(
                (
                    4,
                    self._n_epochs,
                    self._n_chans,
                    len(fband[0]),
                    len(fband[1]),
                ),
                dtype=np.complex128,
            )
            for lfreq_i, lfreq in enumerate(fband[0]):
                for hfreq_i, hfreq in enumerate(fband[1]):
                    coeffs[0, :, :, lfreq_i, hfreq_i] = self._fft_data[
                        ..., lfreq
                    ]
                    coeffs[1, :, :, lfreq_i, hfreq_i] = self._fft_data[
                        ..., hfreq - lfreq
                    ]
                    coeffs[2, :, :, lfreq_i, hfreq_i] = self._fft_data[
                        ..., hfreq
                    ]
                    coeffs[3, :, :, lfreq_i, hfreq_i] = self._fft_data[
                        ..., hfreq + lfreq
                    ]

            fft_coeffs.append(coeffs)

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
        bispectra = []
        for fband_coeffs in fft_coeffs:
            n_lfreqs = fband_coeffs.shape[3]
            n_hfreqs = fband_coeffs.shape[4]
            fband_bispectra = np.zeros(
                (
                    self._n_epochs,
                    2,  # left and right flanking peaks
                    self._n_chans,
                    self._n_chans,
                    self._n_chans,
                    n_lfreqs,
                    n_hfreqs,
                ),
                dtype=np.complex128,
            )
            for lfreq_i in range(n_lfreqs):
                for hfreq_i in range(n_hfreqs):
                    for chan_i, chan in enumerate(self._use_chans):
                        fband_bispectra[
                            :, 0, :, :, chan_i, lfreq_i, hfreq_i
                        ] = (
                            fband_coeffs[0, :, :, lfreq_i, hfreq_i].T
                            @ (
                                fband_coeffs[1, :, :, lfreq_i, hfreq_i].T
                                * fband_coeffs[
                                    2, :, chan, lfreq_i, hfreq_i
                                ].conj()
                            ).T
                        )
                        fband_bispectra[
                            :, 1, :, :, chan_i, lfreq_i, hfreq_i
                        ] = (
                            fband_coeffs[0, :, :, lfreq_i, hfreq_i].T
                            @ (
                                fband_coeffs[2, :, :, lfreq_i, hfreq_i].T
                                * fband_coeffs[
                                    3, :, chan, lfreq_i, hfreq_i
                                ].conj()
                            ).T
                        )

            bispectra.append(fband_bispectra.mean(axis=0))

        return bispectra

    def _compute_bivariance(self) -> None:
        """"""
        self._bivariance_down = self._compute_bivariance_from_bispectra(
            self._bispectra_down
        )
        self._bivariance_up = self._compute_bivariance_from_bispectra(
            self._bispectra_up
        )

    def _compute_bivariance_from_bispectra(self, bispectra) -> None:
        """"""
        bivariance = []
        for fband_bispectra in bispectra:
            n_lfreqs = fband_bispectra.shape[4]
            n_hfreqs = fband_bispectra.shape[5]
            fband_bivariance = np.zeros(
                (len(self._seeds), len(self._n_targets), n_lfreqs, n_hfreqs)
            )
            for lfreq_i in range(n_lfreqs):
                for hfreq_i in range(n_hfreqs):
                    for seed_i, seed in enumerate(self._seeds):
                        for target_i, target in enumerate(self._targets):
                            fband_bivariance[
                                seed_i, target_i, lfreq_i, hfreq_i
                            ] = fband_bispectra[
                                0, seed, target, target, lfreq_i, hfreq_i
                            ]
            bivariance.append(fband_bivariance)

        return bivariance

    def _compute_pac(self, down, up) -> None:
        """"""
        pac = []
        for fband_down, fband_up in zip(down, up):
            n_lfreqs = fband_down.shape[2]
            n_hfreqs = fband_down.shape[3]
            fband_pac = np.zeros(
                len(self._seeds), len(self._targets), n_lfreqs, n_hfreqs
            )
            for seed_i, seed in enumerate(self._seeds):
                for target_i, target in enumerate(self._targets):
                    for lfreq_i in range(n_lfreqs):
                        for hfreq_i in range(n_hfreqs):
                            fband_pac[
                                seed_i, target_i, lfreq_i, hfreq_i
                            ] = np.mean(
                                [
                                    fband_down[seed, target, lfreq_i, hfreq_i],
                                    fband_up[seed, target, lfreq_i, hfreq_i],
                                ]
                            )
            pac.append(fband_pac)

        return pac

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
