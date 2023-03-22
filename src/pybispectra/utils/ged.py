"""Tools for performing generalised eigendecompositions."""

from copy import deepcopy
from warnings import warn

import numpy as np
from mne.decoding import SSD
from mne.time_frequency import csd_array_fourier, csd_array_multitaper
import scipy as sp

from pybispectra.utils.utils import _create_mne_info
from pybispectra.utils._process import _ProcessTimeBase


class SpatioSpectralFilter(_ProcessTimeBase):
    r"""Class for performing spatiospectral filtering.

    Parameters
    ----------
    data : numpy.ndarray, shape [epochs x channels x times]

    sfreq : float
        Sampling frequency of :attr:`data` (in Hz).

    verbose : bool (default True)
        Whether or not to report the progress of the processing.

    Attributes
    ----------
    transformed_data : numpy.ndarray, shape [epochs x :attr:`rank` x times]
        :attr:`data` transformed with :attr:`filters`.

    filters : numpy.ndarray, shape [channels x :attr:`rank`]
        Spatial filters (eigenvectors of the eigendecomposition). Sorted in
        descending order according to the size of the signal:noise ratios of
        :attr:`ratios`.

    patterns : numpy.ndarray, shape [:attr:`rank` x channels]
        Spatial patterns for each of the spatial filters.

    ratios : numpy.ndarray, shape [:attr:`rank`]
        Signal:noise ratios for each of the spatial filters (eigenvalues of the
        eigendecomposition). Sorted in descending order.

    Notes
    -----
    The filtering methods used here rely on generalised eigendecomposition: a
    multivariate method for generating filters that maximise discrimination
    between signal and noise features of the data :footcite:`Cohen2022`. Two
    approaches are available: spatiospectral decomposition (SSD)
    :footcite:`Nikulin2011`; and harmonic power maximisation (HPMax)
    :footcite:`Bartz2019`.

    In SSD, a signal band of interest and a flanking noise band are selected,
    with the goal of generating a spatial filter that will maximise information
    in the signal band of interest, and minimise information from the flanking
    noise frequencies.

    HPMax is an extension of SSD, in which not only the signal band of
    interest, but also information from the harmonics of these frequencies is
    maximised with the spatial filters. This can be beneficial when information
    at the harmonics as well as the base frequencies is also of interest, such
    as with wave shape analyses :footcite:`Bartz2019`. Depending on the level
    of noise in the data, however, performance of HPMax can be detrimental
    compared to SSD :footcite:`Bartz2019`. It is recommended that you explore
    the performance of the different methods on your data.

    Generalised eigendecompositions have the general form

    :math:`NW \Lambda=SW`,

    which can also be represented in the ratio form

    :math:`\Lambda=\frac{W^TSW}{W^TNW}`,

    where :math:`S` and :math:`N` are the covariance matrices for the signal
    and noise information in the data, respectively, :math:`W` is a common set
    of spatial filters (which, when applied to the data, will maximise signal
    information content and minimise noise information content), and
    :math:`\Lambda` is a diagonal matrix of eigenvalues representing the ratio
    of signal:noise information content in the data transformed with each
    spatial filter. Accordingly, spatial filters for which the signal:noise
    ratio is > 1 are generally of interest.

    References
    ----------
    .. footbibliography::
    """

    filters = None
    patterns = None
    ratio = None
    _transformed_data = None

    _fitted = False

    def __init__(
        self,
        data: np.ndarray,
        sfreq: float,
        verbose: bool = True,
    ) -> None:  # noqa D107
        # super call required to check that sfreq is given
        super().__init__(data, freqs, sfreq, verbose)

    def _sort_freq_bounds(
        self,
        signal_bounds: tuple[float],
        noise_bounds: tuple[float],
        signal_noise_gap: float,
    ) -> None:
        """Sort frequency bound inputs."""
        if not isinstance(signal_bounds, tuple) or not all(
            isinstance(entry, float) for entry in signal_bounds
        ):
            raise TypeError("`signal_bounds` must be a tuple of floats.")
        if not isinstance(noise_bounds, tuple) or not all(
            isinstance(entry, float) for entry in noise_bounds
        ):
            raise TypeError("`noise_bounds` must be a tuple of floats.")
        if not isinstance(signal_noise_gap, float):
            raise TypeError("`signal_noise_gap` must be a float.")

        if len(signal_bounds) != 2 or len(noise_bounds) != 2:
            raise ValueError(
                "`signal_bounds` and `noise_bounds` must have lengths of 2."
            )

        if (
            not signal_bounds[0] - signal_noise_gap > noise_bounds[0]
            or not signal_bounds[1] + signal_noise_gap < noise_bounds[1]
        ):
            raise ValueError(
                "The frequencies of `noise_bounds` must lie outside of "
                "`signal_bounds` +/- `signal_noise_gap`."
            )

        self.signal_bounds = deepcopy(signal_bounds)
        self.noise_bounds = deepcopy(noise_bounds)
        self.signal_noise_gap = deepcopy(signal_noise_gap)
        self._n_noise_freqs = [
            self.signal_bounds[0] - self.noise_bounds[0],
            self.noise_bounds[1] - self.signal_bounds[1],
        ]

    def _sort_n_harmonics(self, n_harmonics: int) -> None:
        """Sort harmonic use input."""
        if not isinstance(n_harmonics, int):
            raise TypeError("`n_harmonics` must be an int.")

        if n_harmonics < 0:
            raise ValueError("`n_harmonics` must be >= 0.")

        if (
            self.signal_bounds[1] * (self.n_harmonics + 1)
        ) + self._n_noise_freqs[1] > self.sfreq * 0.5:
            raise ValueError(
                "`n_harmonics` for the requested signal and noise freqs. "
                "extends beyond the Nyquist frequency."
            )

        self.n_harmonics = deepcopy(n_harmonics)

    def _sort_indices(self, indices: tuple[int] | None) -> None:
        """Sort channel indices input."""
        indices = deepcopy(indices)

        if indices is None:
            indices = tuple(np.arange(self._n_chans))

        if not isinstance(indices, tuple) or not all(
            isinstance(entry, int) for entry in indices
        ):
            raise TypeError("`indices` must be a tuple of ints.")

        if np.min(indices) < 0 or np.max(indices) >= self._n_chans:
            raise ValueError(
                "`indices` can only contain channel indices >= 0 or < the "
                "number of channels in the data."
            )

        self.indices = indices

    def _sort_rank(self, rank: int | None) -> None:
        """Sort rank subspace projection input."""
        rank = deepcopy(rank)

        if rank is None:
            rank = deepcopy(self._n_chans)

        if not isinstance(rank, int):
            raise TypeError("`rank` must be an int.")

        if rank < 1 or rank > self._n_chans:
            raise ValueError(
                "`rank` must be >= 1 and <= the number of channels in "
                "`indices`"
            )

        self.rank = rank

    def _sort_csd_method(self, csd_method: str) -> None:
        """Sort CSD computation method input."""
        accepted_methods = ["multitaper", "fourier"]
        if csd_method not in accepted_methods:
            raise ValueError("`csd_method` is not recognised.")

    def fit_transform_ssd(
        self,
        signal_bounds: tuple[float],
        noise_bounds: tuple[float],
        signal_noise_gap: int = 1.0,
        indices: tuple[int] | None = None,
        rank: int | None = None,
    ) -> None:
        """Fit SSD filters and transform the data.

        Parameters
        ----------
        signal_bounds : tuple of float
            Frequencies (in Hz) to treat as the signal of interest.

        noise_bounds : tuple of float
            Frequencies (in Hz) to treat as the noise, excluding the frequency
            boundary in :attr:`signal_bounds` +/- :attr:`signal_noise_gap`.

        signal_noise_gap : float (default 1.0)
            Frequency count (in Hz) to treat as a transtition boundary between
            :attr:`signal_bounds` and :attr:`noise_bounds`. Used to reduce
            spectral leakage between he signal and noise frequencies.

        indices : tuple of int | None (default None)
            Channel indices to fit the filters to. If ``None``, all channels
            are used.

        rank : int | None (default None)
            Rank subspace to project the data to. If ``None``, no projection is
            performed.

        Notes
        -----
        The SSD implementation in MNE is used to compute the filters
        (mne.decoding.SSD).

        References
        ----------
        .. footbibliography::
        """
        self._sort_freq_bounds(signal_bounds, noise_bounds, signal_noise_gap)
        self._sort_indices(indices)
        self._sort_rank(rank)

        info = _create_mne_info(self._n_chans, self.sfreq)
        filt_params_signal, filt_params_noise = self._create_mne_filt_params(
            signal_bounds, noise_bounds, signal_noise_gap
        )

        self._compute_ssd(info, filt_params_signal, filt_params_noise)

        self._fitted = True

    def _create_mne_filt_params(
        self,
        signal_bounds: tuple[float],
        noise_bounds: tuple[float],
        signal_noise_gap: float,
    ) -> tuple[dict, dict]:
        """Create filter parameters for use with MNE's SSD implementation.

        Parameters
        ----------
        signal_bounds : tuple of float

        noise_bounds : tuple of float

        signal_noise_gap : float

        Returns
        -------
        filt_params_signal : dict
            Filter parameters for the signal frequencies, with the keys
            ``"l_freq"``, ``"h_freq"``, ``"l_trans_bandwidth"``, and
            ``"h_trans_bandwidth"``, as in mne.decoding.SSD.

        filt_params_noise : dict
            Filter parameters for the noise frequencies, with the keys
            ``"l_freq"``, ``"h_freq"``, ``"l_trans_bandwidth"``, and
            ``"h_trans_bandwidth"``, as in mne.decoding.SSD.

        Notes
        -----
        ``"l_freq"`` and ``"h_freq"`` are derived from the first and second
        entries, respectively, of :attr:``signal_bounds`` and
        :attr:``noise_bounds``. ``"l_trans_bandwidth"`` and
        ``"h_trans_bandwidth"`` are taken as :attr:``signal_noise_gap`` for
        both signal and noise filters.
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

    def _compute_ssd(self) -> None:
        """Compute SSD on data using the MNE implementation."""
        info = _create_mne_info(self._n_chans, self.sfreq)
        filt_params_signal, filt_params_noise = self._create_mne_filt_params()

        ssd = SSD(
            info,
            filt_params_signal,
            filt_params_noise,
            reg=None,
            n_components=None,
            picks=None,
            sort_by_spectral_ratio=False,
            return_filtered=False,
            rank=self.rank,
        )
        self._transformed_data = ssd.fit_transform(self.data)

        self.filters = ssd.filters_.copy()
        self.patterns = ssd.patterns_.copy()
        self.ratio = ssd.eigvals_.copy()

    def fit_transform_hpmax(
        self,
        signal_bounds: tuple[float],
        noise_bounds: tuple[float],
        n_harmonics: int = 0,
        indices: tuple[int] | None = None,
        rank: int | None = None,
        csd_method: str = "multitaper",
        n_fft: int | None = None,
        mt_bandwidth: float = 5.0,
        mt_adaptive: bool = True,
        mt_low_bias: bool = True,
        n_jobs: int = 1,
    ) -> None:
        """Fit HPMax filters and transform the data.

        Parameters
        ----------
        signal_bounds : tuple of float
            Frequencies (in Hz) to treat as the signal of interest.

        noise_bounds : tuple of float
            Frequencies (in Hz) to treat as the noise, excluding the frequency
            boundary in :attr:`signal_bounds` +/- :attr:`signal_noise_gap`. For
            harmonics, the same number of frequency bins around the harmonic
            frequencies are taken as noise frequencies.

        n_harmonics : int (default 0)
            Number of harmonic frequencies of :attr:`signal_bounds` to use when
            computing the filters. If ``0``, no harmonics are used. Only used
            when :attr:`method` is ``"hpmax"``.

        indices : tuple of int | None (default None)
            Channel indices to fit the filters to. If ``None``, all channels
            are used.

        rank : int | None (default None)
            Rank subspace to project the data to. If ``None``, no projection is
            performed.

        csd_method : str (default ``"multitaper"``)
            Method to use when computing the CSD. Can be ``"multitaper"`` or
            ``"fourier"``.

        n_fft : int | None (default None)
            Number of samples in the FFT. If ``None``, the number of times in
            each epoch is used.

        mt_bandwidth : float (default 5.0)
            Bandwidth of the multitaper windowing function (in Hz). Only used
            if :attr:``csd_method`` is ``"multitaper"``.

        mt_adaptive : bool (default True)
            Whether to use adaptive weights when combining tapered spectra.
            Only used if :attr:``csd_method`` is ``"multitaper"``.

        mt_low_bias : bool (default True)
            Whether to only use tapers with > 90% spectral concentration within
            the bandwidth. Only used if :attr:``csd_method`` is
            ``"multitaper"``.

        n_jobs : int (default 1)
            Number of jobs to use when computing the CSD.

        Notes
        -----
        MNE is used to compute the CSD, from which the covariance matrices are
        obtained :footcite:`Bartz2019 (mne.time_frequency.csd_array_multitaper
        and mne.time_frequency.csd_array_fourier).

        References
        ----------
        .. footbibliography::
        """
        self._sort_freq_bounds(signal_bounds, noise_bounds, 0.0)
        self._sort_n_harmonics(n_harmonics)
        self._sort_indices(indices)
        self._sort_rank(rank)
        self._sort_csd_method(csd_method)

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

    def _compute_csd(
        self,
        csd_method: str,
        n_fft: int | None,
        mt_bandwidth: float,
        mt_adaptive: bool,
        mt_low_bias: bool,
        n_jobs: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute the CSD of the data.

        Returns
        -------
        csd : numpy.ndarray, shape [channels x channels x frequencies]
            CSD of the data.

        freqs : numpy.ndarray, shape [frequencies]
            Frequencies in ``csd``.
        """
        fmin, fmax = self._get_fmin_fmax()

        if csd_method == "multitaper":
            csd = csd_array_multitaper(
                X=self.data,
                sfreq=self.sfreq,
                t0=0,
                fmin=fmin,
                fmax=fmax,
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
                X=self.data,
                sfreq=self.sfreq,
                t0=0,
                fmin=fmin,
                fmax=fmax,
                tmin=None,
                tmax=None,
                ch_names=None,
                n_fft=n_fft,
                projs=None,
                n_jobs=n_jobs,
                verbose=False,
            )

        freqs = csd.frequencies.copy()
        csd = np.array(
            [csd.get_data(freq) for freq in csd.frequencies]
        ).transpose(1, 2, 0)

        return csd, freqs

    def _get_fmin_fmax(self) -> tuple[float, float]:
        """Get minimum and maximum freqs. to compute the CSD for."""
        fmin = self.noise_bounds[0] - self.signal_noise_gap
        fmax = (
            self.signal_bounds[1] * (self.n_harmonics + 1)
        ) + self._n_noise_freqs[1]

        return fmin, fmax

    def _compute_hpmax(self, csd: np.ndarray, freqs: np.ndarray) -> None:
        """Compute HPMax on the data CSD."""
        cov_signal, cov_noise = self._compute_cov_from_csd(csd, freqs)
        cov_signal, cov_noise, projection = self._project_cov_rank_subspace(
            cov_signal, cov_noise
        )

        eigvals, eigvects = sp.linalg.eigh(cov_signal, cov_noise)
        ix = np.argsort(eigvals)[::-1]  # sort in descending order

        self._transformed_data = np.einsum(
            "ijk,jl->ilk", self.data, self.filters
        )

        self.filters = projection @ eigvects[:, ix]  # project to sensor space
        self.patterns = np.linalg.pinv(self.filters)
        self.ratios = eigvals[ix]

    def _compute_cov_from_csd(
        self, csd: np.ndarray, freqs: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute covariance of the signal and noise freqs. from the CSD.

        Returns
        -------
        cov_signal : numpy.ndarray, shape [channels x channels]
            Covariance of the signal frequencies.

        cov_noise : numpy.ndarray, shape [channels x channels]
            Covariance of the noise frequencies.
        """
        csd = np.real(csd)
        freqs = freqs.tolist()

        cov_signal = np.zeros((self._n_chans, self._n_chans), dtype=np.float64)
        cov_noise = np.zeros((self._n_chans, self._n_chans), dtype=np.float64)

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
            cov_noise += (
                np.mean(csd[..., s_hfreq_i + 1 : n_hfreq_i + 1], axis=2) * 0.5
            )

        return cov_signal, cov_noise

    def _project_cov_rank_subspace(
        self, cov_signal: np.ndarray, cov_noise: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Project covariance matrices to their rank subspace.

        Returns
        -------
        cov_signal : numpy.ndarray, shape [:attr:`rank` x :attr:`rank`]
            Covariance of the signal frequencies projected into the rank
            subspace.

        cov_noise : numpy.ndarray, shape [:attr:`rank` x :attr:`rank`]
            Covariance of the noise frequencies projected into the rank
            subspace.

        projection : numpy.ndarray, shape [channels x :attr:`rank`]
            Rank subspace projection matrix.
        """
        if self.rank < self._n_chans:
            eigvals, eigvects = sp.linalg.eigh(cov_signal)
            ix = np.argsort(eigvals)[::-1]  # sort in descending order
            eigvals = eigvals[ix]
            eigvects = eigvects[:, ix]
            projection = eigvects[:, : self.rank] @ (
                np.eye(self.rank) * eigvals[: self.rank] ** -0.5
            )
        else:
            projection = np.eye(self._n_chans)

        cov_signal = projection.T @ cov_signal @ projection
        cov_noise = projection.T @ cov_noise @ projection

        return cov_signal, cov_noise, projection

    @property
    def transformed_data(self, min_ratio: float = 1.0) -> np.ndarray:
        """Return the transformed data.

        Parameters
        ----------
        min_ratio : float (default 1.0)
            Minimum required value of :attr:`ratios` to return the data
            transformed with the corresponding spatial filter.

        Returns
        -------
        transformed_data : numpy.ndarray, shape [epochs x components x times]
            Transformed data with only those components created with filters
            whose signal:noise ratios are > :attr:`min_ratio`.

        Notes
        -----
        Raises a warning if no components have a signal:noise ratio >
        :attr:`min_ratio` and :attr:`verbose` is ``True``.
        """
        if not isinstance(min_ratio, float):
            raise TypeError("`min_ratio` must be a float")

        data = self._transformed_data[:, np.where(self.ratios > min_ratio)[0]]
        if self.verbose and data.shape[1] == 0:
            warn(
                "No signal:noise ratios are greater than the requested "
                "minimum; returning an empty array.",
                UserWarning,
            )

        return data.copy()
