"""Tools for performing generalised eigendecompositions."""

from copy import deepcopy

import numpy as np

from pybispectra.utils._process import _ProcessBase


class SpatioSpectralFilter(_ProcessBase):
    """"""

    filters = None
    patterns = None
    transformed_data = None

    _fitted = False

    def fit_filters(
        self,
        signal_bounds: tuple[float],
        noise_bounds: tuple[float],
        signal_noise_gap: float = 1.0,
        method: str = "narrowband_ssd",
        n_harmonics: int = 0,
        indices: tuple[int] | None = None,
        rank: int | None = None,
    ) -> None:
        """Fit spatiospectral filters to the data.

        Parameters
        ----------
        signal_bounds : tuple of float
            Frequencies (in Hz) to treat as the signal of interest.

        noise_bounds : tuple of float
            Frequencies (in Hz) to treat as the noise, excluding the frequency
            boundary in :attr:`signal_bounds` +/- :attr:`signal_noise_gap`.

        signal_noise_gap : float (default 1.0)
            Frequency boundary (in Hz) to ignore between :attr:`signal_bounds`
            and :attr:`noise_bounds`. Used to minimise spectral leakage between
            the signal and noise frequencies.

        method : str (default ``"narrowband_ssd"``)
            Form of filtering to perform. Accepts: ``"narrowband_ssd"`` -
            spatiospectral decomposition (SSD) applied to the
            narrowband-filtered timeseries signal :footcite:`Nikulin2011`;
            ``"broadband_ssd"`` SSD applied to the broadband-filtered
            timeseries signal (:math:`SSD^-` in :footcite:`Bartz2019`);
            ``"hpmax"`` - harmonic power maximisation :footcite:`Bartz2019`.

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

        References
        ----------
        .. footbibliography::
        """
        self._sort_freq_bounds(signal_bounds, noise_bounds, signal_noise_gap)
        self._sort_method(method)
        self._sort_n_harmonics(n_harmonics)
        self._sort_indices(indices)
        self._sort_rank(rank)

        if method == "hpmax":
            self._compute_hpmax()
        else:
            self._compute_ssd()

        self.patterns = np.linalg.pinv(self.filters).T

        self._fitted = True

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

    def _sort_method(self, method: str) -> None:
        """Sort filter method input."""
        method = deepcopy(method)

        accepted_methods = ["narrowband_ssd", "broadband_ssd", "hpmax"]
        if method not in accepted_methods:
            raise ValueError("`method` is not recognised.")

        self.method = method

    def _sort_n_harmonics(self, n_harmonics: int) -> None:
        """Sort harmonic use input."""
        if self.method == "hpmax":
            if not isinstance(n_harmonics, int):
                raise TypeError("`n_harmonics` must be an int.")

            if n_harmonics < 0:
                raise ValueError("`n_harmonics` must be >= 0.")

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

    def transform_data(self, n_filters: int | None = None) -> None:
        """Transform the data using the spatiospectral filters."""
        if not self._fitted:
            raise ValueError(
                "Filters must be fitted before data can be transformed."
            )

        n_filters = self._sort_n_filters(n_filters)

        self.transformed_data = np.einsum(
            "ijk,jl->ilk", self.data, self.filters[:, n_filters]
        )

    def _sort_n_filters(self, n_filters: int | None) -> int:
        """Sort input for number of filters to transform data with."""
        n_filters = deepcopy(n_filters)

        if n_filters is None:
            n_filters = deepcopy(self._n_chans)

        if not isinstance(n_filters, int):
            raise TypeError("`n_filters` must be an int.")

        if n_filters < 1 or n_filters > self.rank:
            raise ValueError(
                "`n_filters` must be > 0 and <= the rank of `data`"
            )

        return n_filters
