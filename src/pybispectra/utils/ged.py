"""Tools for performing generalised eigendecompositions."""

from multiprocessing import cpu_count
from warnings import warn

import numpy as np
import scipy as sp
from mne import Info
from mne.decoding import SSD
from mne.time_frequency import csd_array_fourier, csd_array_multitaper

from pybispectra.utils._defaults import _precision
from pybispectra.utils._utils import _create_mne_info, _int_like, _number_like
from pybispectra.utils.utils import compute_rank


class SpatioSpectralFilter:
    r"""Class for performing spatiospectral filtering.

    Parameters
    ----------
    data : ~numpy.ndarray, shape of [epochs, channels, times]
        Data to perform spatiospectral filtering on.

    sampling_freq : int | float
        Sampling frequency (in Hz) of :attr:`data`.

    verbose : bool (default True)
        Whether or not to report the progress of the processing.

    Methods
    -------
    fit_hpmax :
        Fit HPMax filters to the data.

    fit_ssd :
        Fit SSD filters to the data.

    transform :
        Transform the data with the fitted filters.

    fit_transform_hpmax :
        Fit HPMax filters and transform the data.

    fit_transform_ssd :
        Fit SSD filters and transform the data.

    get_transformed_data :
        Return the transformed data.

    Attributes
    ----------
    filters : ~numpy.ndarray, shape of [channels, rank]
        Spatial filters (eigenvectors of the eigendecomposition). Sorted in descending
        order according to the size of the signal-to-noise ratios of :attr:`ratios`.

    patterns : ~numpy.ndarray, shape of [rank, channels]
        Spatial patterns for each of the spatial filters.

    ratios : ~numpy.ndarray, shape of [rank]
        Signal-to-noise ratios for each of the spatial filters (eigenvalues of the
        eigendecomposition). Sorted in descending order.

    Notes
    -----
    The filtering methods used here rely on generalised eigendecomposition: a
    multivariate method for generating filters that maximise discrimination between
    signal and noise features of the data :footcite:`Cohen2022`. Two approaches are
    available: spatio-spectral decomposition (SSD) :footcite:`Nikulin2011`; and harmonic
    power maximisation (HPMax) :footcite:`Bartz2019`.

    In SSD, a signal band of interest and a flanking noise band are selected, with the
    goal of generating a spatial filter that will maximise information in the signal
    band of interest, and minimise information from the flanking noise frequencies. In
    practice, this is implemented by bandpass filtering the data at a frequency range of
    interest and computing the covariance matrices on the signal and noise periods which
    are passed to the generalised eigendecomposition.

    HPMax is an extension of SSD, in which not only the signal band of interest, but
    also information from the harmonics of these frequencies is maximised with the
    spatial filters. HPMax is implemented by computing the cross-spectral density of the
    data, which can then be summed across the signal and noise frequencies of interest,
    taking advantage of the fact that the cross-spectral density is a frequency-resolved
    representation of the covariance matrix.

    Depending on the signal-to-noise ratio (SNR) of the data, the performance of these
    methods for recovering the underlying signal of interest - measured by the ability
    to suppress noise whilst retaining the original waveshape - can vary
    :footcite:`Bartz2019`:

    *   Low SNRs: SSD filters applied to bandpass-filtered data (SSD+) can show
        favourable performance compared to SSD applied to broadband data (SSD-) and
        HPMax. The distortion of waveshape with bandpass filtering in SSD+ is
        compensated for by the higher level of noise reduction.

    *   Intermediate SNRs: SSD- and HPMax performance can match or surpass SSD+
        performance, and HPMax can outperform SSD-. The lower level of noise in the base
        signal means that the weaker degree of noise reduction with SSD- and HPMax is
        sufficient to uncover the signal without the distorting effects of bandpass
        filtering on waveshape.

    *   High SNRs: HPMax can show increased or similar performance to SSD-.

    It is therefore recommended that you explore the performance of the different
    methods on your data.

    Generalised eigendecompositions have the general form

    :math:`\textbf{NW} \boldsymbol{\Lambda}=\textbf{SW}` ,

    which can also be represented in the ratio form

    :math:`\boldsymbol{\Lambda}=\Large{\frac{\textbf{W}^T\textbf{SW}}{
    \textbf{W}^T\textbf{NW}}}` ,

    where :math:`\textbf{S}` and :math:`\textbf{N}` are the covariance matrices for the
    signal and noise information in the data, respectively; :math:`\textbf{W}` is a
    common set of spatial filters (which, when applied to the data, will maximise signal
    information content and minimise noise information content); and
    :math:`\boldsymbol{\Lambda}` is a diagonal matrix of eigenvalues representing the
    ratio of signal-to-noise information content in the data transformed with each
    spatial filter. Accordingly, spatial filters for with an SNR > 1 are generally of
    interest.

    References
    ----------
    .. footbibliography::
    """

    data = None
    _n_epochs = None
    _n_chans = None
    _n_times = None

    sampling_freq = None

    indices = None
    _use_n_chans = None

    signal_bounds = None
    noise_bounds = None
    signal_noise_gap = None
    _n_noise_freqs = None

    n_harmonics = None

    bandpass_filter = None

    rank = None

    filters = None
    patterns = None
    ratios = None
    _ssd = None
    _transformed_data = None

    _fitted = False
    _fitted_method = None
    _transformed = False

    def __init__(
        self,
        data: np.ndarray,
        sampling_freq: int | float,
        verbose: bool = True,
    ) -> None:  # noqa: D107
        self.verbose = verbose
        self._sort_init_inputs(data, sampling_freq)

    def _sort_init_inputs(self, data: np.ndarray, sampling_freq: float) -> None:
        """Check init. inputs are appropriate."""
        if not isinstance(data, np.ndarray):
            raise TypeError("`data` must be a NumPy array.")
        if data.ndim != 3:
            raise ValueError("`data` must be a 3D array.")

        if not isinstance(sampling_freq, _number_like):
            raise TypeError("`sampling_freq` must be an int or a float.")
        self.sampling_freq = sampling_freq

        self._n_epochs, self._n_chans, self._n_times = data.shape

        self.data = np.asarray(data, dtype=_precision.real)

    def _sort_freq_bounds(
        self,
        signal_bounds: tuple[int | float],
        noise_bounds: tuple[int | float],
        signal_noise_gap: int | float,
    ) -> None:
        """Sort frequency bound inputs."""
        if not isinstance(signal_bounds, tuple) or not all(
            isinstance(entry, _number_like) for entry in signal_bounds
        ):
            raise TypeError("`signal_bounds` must be a tuple of ints or floats.")
        if not isinstance(noise_bounds, tuple) or not all(
            isinstance(entry, _number_like) for entry in noise_bounds
        ):
            raise TypeError("`noise_bounds` must be a tuple of ints or floats.")
        if not isinstance(signal_noise_gap, _number_like):
            raise TypeError("`signal_noise_gap` must be an int or a float.")

        if len(signal_bounds) != 2 or len(noise_bounds) != 2:
            raise ValueError(
                "`signal_bounds` and `noise_bounds` must have lengths of 2."
            )

        if (
            not signal_bounds[0] - signal_noise_gap > noise_bounds[0]
            or not signal_bounds[1] + signal_noise_gap < noise_bounds[1]
        ):
            raise ValueError(
                "The frequencies of `noise_bounds` must lie outside of `signal_bounds` "
                "+/- `signal_noise_gap`."
            )

        self.signal_bounds = signal_bounds
        self.noise_bounds = noise_bounds
        self.signal_noise_gap = signal_noise_gap
        self._n_noise_freqs = [
            self.signal_bounds[0] - self.noise_bounds[0],
            self.noise_bounds[1] - self.signal_bounds[1],
        ]

    def _sort_bandpass_filter(self, bandpass_filter: bool) -> None:
        """Sort bandpass filter input."""
        if not isinstance(bandpass_filter, bool):
            raise TypeError("`bandpass_filter` must be a bool.")

        self.bandpass_filter = bandpass_filter

    def _sort_n_harmonics(self, n_harmonics: int) -> None:
        """Sort harmonic use input."""
        if not isinstance(n_harmonics, _int_like):
            raise TypeError("`n_harmonics` must be an int.")

        if n_harmonics < -1:
            raise ValueError("`n_harmonics` must be >= -1.")

        if n_harmonics == -1:
            n_harmonics = (
                int(
                    (self.sampling_freq * 0.5 - self._n_noise_freqs[1])
                    // self.signal_bounds[1]
                )
                - 1
            )

        if (self.signal_bounds[1] * (n_harmonics + 1)) + self._n_noise_freqs[
            1
        ] > self.sampling_freq * 0.5:
            raise ValueError(
                "`n_harmonics` for the requested signal and noise frequencies extends "
                "beyond the Nyquist frequency."
            )

        self.n_harmonics = n_harmonics

    def _sort_indices(self, indices: tuple[int] | None) -> None:
        """Sort channel indices input."""
        if indices is None:
            indices = tuple(np.arange(self._n_chans, dtype=np.int32).tolist())

        if not isinstance(indices, tuple) or not all(
            isinstance(entry, _int_like) for entry in indices
        ):
            raise TypeError("`indices` must be a tuple of ints.")

        if min(indices) < 0 or max(indices) >= self._n_chans:
            raise ValueError(
                "`indices` can only contain channel indices >= 0 or < the number of "
                "channels in the data."
            )

        self.indices = indices
        self._use_n_chans = len(indices)

    def _sort_rank(self, rank: int | None) -> None:
        """Sort rank subspace projection input."""
        if rank is None:
            rank = compute_rank(self.data)

        if not isinstance(rank, _int_like):
            raise TypeError("`rank` must be an int.")

        if rank < 1 or rank > self._use_n_chans:
            raise ValueError(
                "`rank` must be >= 1 and <= the number of channels in `indices`"
            )

        self.rank = rank

    def _sort_csd_method(self, csd_method: str) -> None:
        """Sort CSD computation method input."""
        accepted_methods = ["multitaper", "fourier"]
        if csd_method not in accepted_methods:
            raise ValueError("`csd_method` is not recognised.")

    def fit_ssd(
        self,
        signal_bounds: tuple[int | float],
        noise_bounds: tuple[int | float],
        signal_noise_gap: int | float = 1.0,
        bandpass_filter: bool = False,
        indices: tuple[int] | None = None,
        rank: int | None = None,
    ) -> None:
        """Fit SSD filters to the data.

        Parameters
        ----------
        signal_bounds : tuple of int or float, length of 2
            Lower and upper frequencies (in Hz), respectively, to treat as the signal of
            interest.

        noise_bounds : tuple of int or float, length of 2
            Lower and upper frequencies (in Hz) to treat as the noise, excluding the
            frequencies in :attr:`signal_bounds`.

        signal_noise_gap : int | float (default ``1.0``)
            Frequency count (in Hz) to treat as a transition boundary between
            :attr:`signal_bounds` and :attr:`noise_bounds`. Used to reduce spectral
            leakage between the signal and noise frequencies.

        bandpass_filter : bool (default False)
            Whether or not to bandpass filter the data before transforming with the SSD
            filters.

        indices : tuple of int | None (default None)
            Channel indices to fit the filters to. If :obj:`None`, all channels are
            used.

        rank : int | None (default None)
            Rank subspace to project the data to. If :obj:`None`, the rank of the data
            is automatically computed and projected to.

        Notes
        -----
        The SSD implementation in MNE is used to compute the filters
        (:class:`mne.decoding.SSD`).

        .. versionadded:: 1.2
        """
        self._sort_freq_bounds(signal_bounds, noise_bounds, signal_noise_gap)
        self._sort_bandpass_filter(bandpass_filter)
        self._sort_indices(indices)
        self._sort_rank(rank)

        if self.verbose:
            print("Fitting SSD filters...\n")

        info = _create_mne_info(self._use_n_chans, self.sampling_freq)
        filt_params_signal, filt_params_noise = self._create_mne_filt_params(
            signal_bounds, noise_bounds, signal_noise_gap
        )

        self._compute_ssd(info, filt_params_signal, filt_params_noise)

        self._fitted = True
        self._fitted_method = "SSD"

        if self.verbose:
            print("    ... SSD filter fitting finished\n")

    def _create_mne_filt_params(
        self,
        signal_bounds: tuple[int | float],
        noise_bounds: tuple[int | float],
        signal_noise_gap: int | float,
    ) -> tuple[dict, dict]:
        """Create filter parameters for use with MNE's SSD implementation.

        Returns
        -------
        filt_params_signal : dict
            Filter parameters for the signal frequencies, with the keys "l_freq",
            "h_freq", "l_trans_bandwidth", and "h_trans_bandwidth", as in
            ``mne.decoding.SSD``.

        filt_params_noise : dict
            Filter parameters for the noise frequencies, with the keys "l_freq",
            "h_freq", "l_trans_bandwidth", and "h_trans_bandwidth", as in
            ``mne.decoding.SSD``.

        Notes
        -----
        "l_freq" and "h_freq" are derived from the first and second entries,
        respectively, of ``signal_bounds`` and ``noise_bounds``. "l_trans_bandwidth" and
        "h_trans_bandwidth" are taken as ``signal_noise_gap`` for both signal and noise
        filters.
        """
        filt_params_signal = {
            "l_freq": signal_bounds[0],
            "h_freq": signal_bounds[1],
            "l_trans_bandwidth": signal_noise_gap,
            "h_trans_bandwidth": signal_noise_gap,
        }

        filt_params_noise = {
            "l_freq": noise_bounds[0],
            "h_freq": noise_bounds[1],
            "l_trans_bandwidth": signal_noise_gap,
            "h_trans_bandwidth": signal_noise_gap,
        }

        return filt_params_signal, filt_params_noise

    def _compute_ssd(
        self, info: Info, filt_params_signal: dict, filt_params_noise: dict
    ) -> None:
        """Compute SSD on data using the MNE implementation."""
        assert all(ch_type == "eeg" for ch_type in info.get_channel_types()), (
            "PyBispectra Internal Error: channel types in `info` should all be 'eeg'. "
            "Please contact the PyBispectra developers."
        )
        self._ssd = SSD(
            info,
            filt_params_signal,
            filt_params_noise,
            reg=None,
            n_components=None,
            picks=None,
            sort_by_spectral_ratio=False,
            return_filtered=self.bandpass_filter,
            rank={"eeg": self.rank},
        )
        self._ssd.fit(self.data[:, self.indices])

        self.filters = self._ssd.filters_
        self.patterns = self._ssd.patterns_
        self.ratios = self._ssd.eigvals_

    def fit_hpmax(
        self,
        signal_bounds: tuple[int | float],
        noise_bounds: tuple[int | float],
        n_harmonics: int = -1,
        indices: tuple[int] | None = None,
        rank: int | None = None,
        csd_method: str = "multitaper",
        n_fft: int | None = None,
        mt_bandwidth: int | float = 5.0,
        mt_adaptive: bool = True,
        mt_low_bias: bool = True,
        n_jobs: int = 1,
    ) -> None:
        """Fit HPMax filters to the data.

        Parameters
        ----------
        signal_bounds : tuple of int or float, length of 2
            Lower and upper frequencies (in Hz), respectively, to treat as the signal of
            interest.

        noise_bounds : tuple of int or float, length of 2
            Lower and upper frequencies (in Hz) to treat as the noise, excluding the
            frequencies in :attr:`signal_bounds`. For harmonics, the same number of
            frequency bins around the harmonic frequencies are taken as noise
            frequencies.

        n_harmonics : int (default ``-1``)
            Number of harmonic frequencies of :attr:`signal_bounds` to use when
            computing the filters. If ``0``, no harmonics are used. If ``-1``, all
            harmonics are used.

        indices : tuple of int | None (default None)
            Channel indices to fit the filters to. If :obj:`None`, all channels are
            used.

        rank : int | None (default None)
            Rank subspace to project the data to. If :obj:`None`, the rank of the data
            is automatically computed and projected to.

        csd_method : str (default ``"multitaper"``)
            Method to use when computing the CSD. Can be ``"multitaper"`` or
            ``"fourier"``.

        n_fft : int | None (default None)
            Number of samples in the FFT. If :obj:`None`, the number of times in each
            epoch is used.

        mt_bandwidth : int | float (default ``5.0``)
            Bandwidth of the multitaper windowing function (in Hz). Only used if
            :attr:`csd_method` is ``"multitaper"``.

        mt_adaptive : bool (default True)
            Whether to use adaptive weights when combining tapered spectra. Only used
            if :attr:`csd_method` is ``"multitaper"``.

        mt_low_bias : bool (default True)
            Whether to only use tapers with > 90% spectral concentration within the
            bandwidth. Only used if :attr:`csd_method` is ``"multitaper"``.

        n_jobs : int (default ``1``)
            Number of jobs to use when computing the CSD. If ``-1``, all available CPUs
            are used.

        Notes
        -----
        MNE is used to compute the CSD, from which the covariance matrices are obtained
        :footcite:`Bartz2019` (:func:`mne.time_frequency.csd_array_multitaper` and
        :func:`mne.time_frequency.csd_array_fourier`).

        .. versionadded:: 1.2
        """
        self._sort_freq_bounds(signal_bounds, noise_bounds, 0.0)
        self._sort_n_harmonics(n_harmonics)
        self._sort_indices(indices)
        self._sort_rank(rank)
        self._sort_csd_method(csd_method)
        n_jobs = self._sort_parallelisation(n_jobs)

        if self.verbose:
            print("Fitting HPMax filters...\n")

        csd, freqs = self._compute_csd(
            csd_method,
            n_fft,
            mt_bandwidth,
            mt_adaptive,
            mt_low_bias,
            n_jobs,
        )
        self._compute_hpmax(csd, freqs)

        self._fitted = True
        self._fitted_method = "HPMax"

        if self.verbose:
            print("    ... HPMax filter fitting finished\n")

    def _sort_parallelisation(self, n_jobs: int) -> int:
        """Sort parallelisation inputs."""
        if not isinstance(n_jobs, _int_like):
            raise TypeError("`n_jobs` must be an integer.")
        if n_jobs < 1 and n_jobs != -1:
            raise ValueError("`n_jobs` must be >= 1 or -1.")
        if n_jobs == -1:
            n_jobs = cpu_count()

        return n_jobs

    def _compute_csd(
        self,
        csd_method: str,
        n_fft: int | None,
        mt_bandwidth: int | float,
        mt_adaptive: bool,
        mt_low_bias: bool,
        n_jobs: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute the CSD of the data.

        Returns
        -------
        csd : numpy.ndarray, shape of [channels, channels, frequencies]
            CSD of the data.

        freqs : numpy.ndarray, shape of [frequencies]
            Frequencies (in Hz) in ``csd``.
        """
        if self.verbose:
            print("    Computing CSD...")

        fmin, fmax = self._get_fmin_fmax()

        if csd_method == "multitaper":
            csd = csd_array_multitaper(
                X=self.data[:, self.indices],
                sfreq=self.sampling_freq,
                t0=0,
                fmin=np.max((fmin - 1, 0)),
                fmax=np.min((fmax + 1, self.sampling_freq)),
                tmin=None,
                tmax=None,
                ch_names=None,
                n_fft=n_fft,
                bandwidth=mt_bandwidth,
                adaptive=mt_adaptive,
                low_bias=mt_low_bias,
                projs=None,
                n_jobs=n_jobs,
                verbose=False,
            )
        else:
            csd = csd_array_fourier(
                X=self.data[:, self.indices],
                sfreq=self.sampling_freq,
                t0=0,
                fmin=np.max((fmin - 1, 0)),
                fmax=np.min((fmax + 1, self.sampling_freq)),
                tmin=None,
                tmax=None,
                ch_names=None,
                n_fft=n_fft,
                projs=None,
                n_jobs=n_jobs,
                verbose=False,
            )

        freqs = csd.frequencies
        csd = np.array(
            [csd.get_data(freq) for freq in freqs], dtype=_precision.complex
        ).transpose(1, 2, 0)

        if self.verbose:
            print("        ... CSD computation finished\n")

        return csd, freqs

    def _get_fmin_fmax(self) -> tuple[float, float]:
        """Get minimum and maximum freqs. to compute the CSD for."""
        fmin = self.noise_bounds[0] - self.signal_noise_gap
        fmax = (self.signal_bounds[1] * (self.n_harmonics + 1)) + self._n_noise_freqs[1]

        return fmin, fmax

    def _compute_hpmax(self, csd: np.ndarray, freqs: np.ndarray) -> None:
        """Compute HPMax on the data CSD."""
        if self.verbose:
            print("    Computing HPMax filters...")

        cov_signal, cov_noise = self._compute_cov_from_csd(csd, freqs)
        cov_signal, cov_noise, projection = self._project_cov_rank_subspace(
            cov_signal, cov_noise
        )

        eigvals, eigvects = sp.linalg.eigh(cov_signal, cov_noise)
        eig_idx = np.argsort(eigvals)[::-1]  # sort in descending order

        self.filters = (projection @ eigvects[:, eig_idx]).astype(_precision.real)
        self.patterns = np.linalg.pinv(self.filters).astype(_precision.real)
        self.ratios = eigvals[eig_idx].astype(_precision.real)

        if self.verbose:
            print("        ... HPMax filter computation finished\n")

    def _compute_cov_from_csd(
        self, csd: np.ndarray, freqs: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute covariance of the signal and noise freqs. from the CSD.

        Returns
        -------
        cov_signal : numpy.ndarray, shape of [channels, channels]
            Covariance of the signal frequencies.

        cov_noise : numpy.ndarray, shape of [channels, channels]
            Covariance of the noise frequencies.
        """
        csd = np.real(csd)
        freqs = freqs.tolist()

        cov_signal = np.zeros(
            (self._use_n_chans, self._use_n_chans), dtype=_precision.real
        )
        cov_noise = np.zeros(
            (self._use_n_chans, self._use_n_chans), dtype=_precision.real
        )

        for harmonic_i in range(self.n_harmonics + 1):
            # signal CSD info.
            s_lfreq = self.signal_bounds[0] * (harmonic_i + 1)
            s_hfreq = self.signal_bounds[1] * (harmonic_i + 1)
            s_lfreq_i = freqs.index(s_lfreq)
            s_hfreq_i = freqs.index(s_hfreq)
            cov_signal += np.mean(csd[..., s_lfreq_i : s_hfreq_i + 1], axis=2)

            # noise CSD info.
            n_lfreq_i = freqs.index(s_lfreq - self._n_noise_freqs[0])
            n_hfreq_i = freqs.index(s_hfreq + self._n_noise_freqs[1])
            cov_noise += np.mean(csd[..., n_lfreq_i:s_lfreq_i], axis=2) * 0.5
            cov_noise += np.mean(csd[..., s_hfreq_i + 1 : n_hfreq_i + 1], axis=2) * 0.5

        return cov_signal, cov_noise

    def _project_cov_rank_subspace(
        self, cov_signal: np.ndarray, cov_noise: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Project covariance matrices to their rank subspace.

        Returns
        -------
        cov_signal : numpy.ndarray, shape of [rank, rank]
            Covariance of the signal frequencies projected into the rank subspace.

        cov_noise : numpy.ndarray, shape of [rank, rank]
            Covariance of the noise frequencies projected into the rank subspace.

        projection : numpy.ndarray, shape of [channels, rank]
            Rank subspace projection matrix.
        """
        if self.rank < self._use_n_chans:
            eigvals, eigvects = sp.linalg.eigh(cov_signal)
            eig_idx = np.argsort(eigvals)[::-1]  # sort in descending order
            eigvals = eigvals[eig_idx]
            eigvects = eigvects[:, eig_idx]
            projection = eigvects[:, : self.rank] @ (
                np.eye(self.rank) * eigvals[: self.rank] ** -0.5
            )
        else:
            projection = np.eye(self._use_n_chans)

        cov_signal = projection.T @ cov_signal @ projection
        cov_noise = projection.T @ cov_noise @ projection

        return cov_signal, cov_noise, projection

    def transform(self, data: np.ndarray | None = None) -> np.ndarray:
        """Transform the data with the fitted filters.

        Parameters
        ----------
        data : ~numpy.ndarray, shape of [epochs, channels, times] | None (default None)
            Data to transform with the fitted filters. If :obj:`None`, the data used to
            fit the filters is transformed.

        Returns
        -------
        transformed_data : ~numpy.ndarray, shape of [epochs, components, times]
            Transformed data.

        Notes
        -----

        .. versionadded:: 1.2
        """
        if not self._fitted:
            raise ValueError(
                "No filters have been fit. Please call `fit_ssd` or `fit_hpmax` before "
                "transforming the data."
            )

        if data is None:
            data = self.data
        if not isinstance(data, np.ndarray):
            raise TypeError("`data` must be a NumPy array.")
        if data.ndim != 3:
            raise ValueError("`data` must be a 3D array.")
        if data.shape[1] != self.filters.shape[0]:
            raise ValueError(
                "`data` must have the same number of channels as the filters."
            )

        if self.verbose:
            print("Transforming data with filters...\n")

        if self.bandpass_filter and self._fitted_method == "SSD":
            self._transformed_data = self._ssd.transform(data)
        else:
            self._transformed_data = np.einsum(
                "ijk,jl->ilk",
                data[:, self.indices],
                self.filters,
                dtype=_precision.real,
            )

        if self.verbose:
            print("    ... Data transformation finished\n")

        self._transformed = True

        return self._transformed_data

    def fit_transform_ssd(self, *args: tuple, **kwargs: dict) -> np.ndarray:
        """Fit SSD filters and transform the data.

        Parameters
        ----------
        args : tuple
            Positional parameters to pass to :meth:`fit_ssd`.

        kwargs : dict
            Keyword parameters to pass to :meth:`fit_ssd`.

        Returns
        -------
        transformed_data : ~numpy.ndarray, shape of [epochs, components, times]
            Transformed data.

        Notes
        -----
        Equivalent to calling :meth:`fit_ssd` followed by :meth:`transform`.
        """
        self.fit_ssd(*args, **kwargs)
        return self.transform()

    def fit_transform_hpmax(self, *args: tuple, **kwargs: dict) -> np.ndarray:
        """Fit HPMax filters and transform the data.

        Parameters
        ----------
        args : tuple
            Positional parameters to pass to :meth:`fit_hpmax`.

        kwargs : dict
            Keyword parameters to pass to :meth:`fit_hpmax`.

        Returns
        -------
        transformed_data : ~numpy.ndarray, shape of [epochs, components, times]
            Transformed data.

        Notes
        -----
        Equivalent to calling :meth:`fit_hpmax` followed by :meth:`transform`.
        """
        self.fit_hpmax(*args, **kwargs)
        return self.transform()

    def get_transformed_data(
        self, min_ratio: int | float = -np.inf, copy: bool = True
    ) -> np.ndarray:
        """Return the transformed data.

        Parameters
        ----------
        min_ratio : int | float (default - numpy.inf)
            Only returns the transformed data for those spatial filters whose
            :attr:`ratios` values is greater or equal to this value.

            .. versionchanged:: 1.2
               Default value changed from ``1.0`` to ``-inf``.

        copy : bool (default True)
            Whether or not to return a copy of the data.

            .. versionadded:: 1.2

        Returns
        -------
        transformed_data : ~numpy.ndarray, shape of [epochs, components, times]
            Transformed data with only those components created with filters whose
            signal-to-noise ratios are > ``min_ratio``.

        Notes
        -----
        Raises a warning if no components have a signal-to-noise ratio > ``min_ratio``
        and :attr:`verbose` is :obj:`True`.
        """
        if not self._transformed:
            raise ValueError(
                "No data has been transformed. Please call `transform`, "
                "`fit_transform_ssd`, or `fit_transform_hpmax` before getting the "
                "transformed data."
            )

        if not isinstance(min_ratio, _number_like):
            raise TypeError("`min_ratio` must be an int or a float")
        if not isinstance(copy, bool):
            raise TypeError("`copy` must be a bool.")

        data = self._transformed_data[:, np.where(self.ratios >= min_ratio)[0]]
        if self.verbose and data.size == 0:
            warn(
                "No signal-to-noise ratios are greater than the requested minimum; "
                "returning an empty array.",
                UserWarning,
            )

        if copy:
            return data.copy()
        return data
