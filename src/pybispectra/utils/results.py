"""Helper tools for storing results."""

from abc import ABC, abstractmethod

import numpy as np
from matplotlib.figure import Figure

from pybispectra.utils._plot import _PlotCFC, _PlotGeneral, _PlotTDE, _PlotWaveShape
from pybispectra.utils._utils import _int_like


class _ResultsBase(ABC):
    """Base class for storing results."""

    f1s: np.ndarray = None
    f2s: np.ndarray = None

    times: np.ndarray = None

    indices: tuple[tuple[int]] = None
    n_nodes: int = None
    _seeds: tuple[int] = None
    _targets: tuple[int] = None
    _kmn: tuple[tuple[int]] = None
    _n_chans: int = None

    def __init__(
        self,
        data: np.ndarray,
        data_ndim: int,
        name: str,
    ) -> None:
        if not isinstance(data, np.ndarray):
            raise TypeError("`data` must be a NumPy array.")
        if data.ndim not in data_ndim:
            raise ValueError(
                "`data` must be a "
                f"{' or '.join([(str(dim) + 'D') for dim in data_ndim])} array."
            )
        self._data = data
        self.shape = data.shape

        if not isinstance(name, str):
            raise TypeError("`name` must be a string.")
        self.name = name

    @abstractmethod
    def _sort_init_inputs(self) -> None:
        """Sort inputs to the object."""

    def _sort_freq_inputs(self, f1s: np.ndarray, f2s: np.ndarray) -> None:
        """Sort ``f1s`` and ``f2s`` inputs."""
        if not isinstance(f1s, np.ndarray) or not isinstance(f2s, np.ndarray):
            raise TypeError("`f1s` and `f2s` must be NumPy arrays.")
        if f1s.ndim != 1 or f2s.ndim != 1:
            raise ValueError("`f1s` and `f2s` must be 1D arrays.")
        self.f1s = f1s
        self.f2s = f2s

    def _sort_times(self, times: np.ndarray | None) -> None:
        """Sort ``times`` input."""
        if self._data.ndim == 3:  # times dimension absent
            return

        if times is None:
            raise ValueError("`times` must be provided for time-resolved results.")
        if not isinstance(times, np.ndarray):
            raise TypeError("`times` must be a NumPy array.")
        if times.ndim != 1:
            raise ValueError("`times` must be a 1D array.")

        self.times = times

    def _check_data_shape(self) -> None:
        """Check that ``data`` has the expected shape."""
        expected_shape = (self.n_nodes, self.f1s.size, self.f2s.size)
        expected_dims = "[nodes, f1s, f2s"
        if self.times is not None:
            expected_shape += (self.times.size,)
            expected_dims += ", times"
        expected_dims += "]"

        if self._data.shape != expected_shape:
            raise ValueError(f"`data` must have shape {expected_dims}.")

    def _sort_indices_seeds_targets(self, indices: tuple[tuple[int]]) -> None:
        """Sort ``indices`` inputs with format ([seeds], [targets])."""
        if not isinstance(indices, tuple):
            raise TypeError("`indices` must be a tuple.")
        if len(indices) != 2:
            raise ValueError("`indices` must have length of 2.")

        seeds = indices[0]
        targets = indices[1]
        for group_idcs in (seeds, targets):
            if not isinstance(group_idcs, tuple):
                raise TypeError("Entries of `indices` must be tuples.")
            if any(not isinstance(idx, _int_like) for idx in group_idcs):
                raise TypeError(
                    "Entries for seeds and targets in `indices` must be ints."
                )
            if any(idx < 0 for idx in group_idcs):
                raise ValueError(
                    "Entries for seeds and targets in `indices` must be >= 0."
                )
        if len(seeds) != len(targets):
            raise ValueError("Entries of `indices` must have equal length.")
        self._n_chans = len(np.unique([*seeds, *targets]))
        self._seeds, self._targets = self._remap_indices_groups(indices)
        self.n_nodes = len(seeds)
        self.indices = indices

    def _sort_indices_channels(self, indices: tuple[int]) -> None:
        """Sort ``indices`` with inputs format [channels]."""
        if not isinstance(indices, tuple):
            raise TypeError("`indices` must be a tuple.")
        if not all(isinstance(idx, _int_like) for idx in indices):
            raise TypeError("Entries of `indices` must be ints.")
        if any(idx < 0 for idx in indices):
            raise ValueError("Entries of `indices` must be >= 0.")
        self._n_chans = len(np.unique(indices))
        self.n_nodes = len(indices)
        self.indices = indices

    def _sort_indices_kmn(self, indices: tuple[tuple[int]]) -> None:
        """Sort ``indices`` inputs with format ([k], [m], [n])."""
        if not isinstance(indices, tuple):
            raise TypeError("`indices` must be a tuple.")
        if len(indices) != 3:
            raise ValueError("`indices` must have length of 3.")

        for group_idcs in indices:
            if not isinstance(group_idcs, tuple):
                raise TypeError("Entries of `indices` must be tuples.")
            if any(not isinstance(idx, _int_like) for idx in group_idcs):
                raise TypeError("Entries for groups in `indices` must be ints.")
            if any(idx < 0 for idx in group_idcs):
                raise ValueError(r"Entries for groups in `indices` must be >= 0.")
        if len(np.unique([len(group) for group in indices])) != 1:
            raise ValueError("Entries of `indices` must have equal length.")
        self._n_chans = len(np.unique(np.ravel(indices)))
        self._kmn = self._remap_indices_groups(indices)
        self.n_nodes = len(indices[0])
        self.indices = indices

    def _remap_indices_groups(self, indices: tuple[tuple[int]]) -> tuple[tuple[int]]:
        """Remap groups of indices (seeds/targets; kmn) to range from 0 to n_chans."""
        # FIXME: This is really ugly. Replace with `np.unique(np.r_[*indices])`` when
        # support for Python 3.10 dropped.
        if len(indices) == 2:
            signal_indices = np.unique(np.r_[indices[0], indices[1]])
        else:
            assert len(indices) == 3, (
                "The number of groups in `indices` is not as expected. Please contact "
                "the PyBispectra developers."
            )
            signal_indices = np.unique(np.r_[indices[0], indices[1], indices[2]])

        return tuple(tuple(np.searchsorted(signal_indices, group)) for group in indices)

    def get_results(
        self, form: str = "raveled", copy: bool = True
    ) -> np.ndarray | tuple[np.ndarray, tuple[tuple[int]]]:
        """Return the results.

        Parameters
        ----------
        form : ``"raveled"`` | ``"compact"`` (default ``"raveled"``)
            How the results should be returned: ``"raveled"`` - results have shape
            ``[nodes, ...]``; ``"compact"`` - results have shape ``[seeds, targets,
            ...]``, where ``...`` represents the data dimensions (e.g. frequencies,
            times).

        copy : bool (default True)
            Whether or not to return a copy of the results.

            .. versionadded:: 1.2

        Returns
        -------
        results : ~numpy.ndarray
            The results.

        indices : tuple of tuple of int, length of 2
            Channel indices of the seeds and targets in ``results``, according to the
            node order in the original data indices. Only returned if ``form`` is
            ``"compact"``.
        """
        accepted_forms = ["raveled", "compact"]
        if form not in accepted_forms:
            raise ValueError("`form` is not recognised.")
        if not isinstance(copy, bool):
            raise TypeError("`copy` must be a bool.")

        if form == "raveled":
            results = self._data
        else:
            results, indices = self._get_compact_results_child()

        if copy:
            results = results.copy()

        if form == "raveled":
            return results
        return results, indices

    def _get_compact_results_child(self) -> tuple:
        """Return a compacted form of the results."""

    def _get_compact_results_parent(
        self, compact_results: np.ndarray
    ) -> tuple[np.ndarray, tuple[tuple[int]]]:
        """Return a compacted form of the results.

        Parameters
        ----------
        compact_results : numpy.ndarray
            Empty results array with shape ``[seeds, targets, ...]``, where ``...``
            represents the data dimensions (e.g. frequencies, times).

        Returns
        -------
        compact_results : numpy.ndarray
            Results array with shape ``[seeds, targets, ...]``, where ``...`` represents
            the data dimensions (e.g. frequencies, times).

        indices : tuple of tuple of int
            Channel indices of ``compact_results`` for the seeds and targets,
            respectively.
        """
        for con_result, seed, target in zip(self._data, self._seeds, self._targets):
            compact_results[seed, target] = con_result

        return compact_results, (self._seeds, self._targets)


class ResultsCFC(_ResultsBase):
    """Class for storing cross-frequency coupling (CFC) results.

    Parameters
    ----------
    data : ~numpy.ndarray, shape of [nodes, low frequencies, high frequencies (, times)]
        Results to store.

    indices : tuple of tuple of int, length of 2
        Indices of the channels for each connection of the results. Should contain two
        tuples of equal length for the seed and target indices, respectively.

    f1s : ~numpy.ndarray, shape of [low frequencies]
        Low frequencies (in Hz) in the results.

    f2s : ~numpy.ndarray, shape of [high frequencies]
        High frequencies (in Hz) in the results.

    times : ~numpy.ndarray, shape of [times] | None (default None)
        Timepoints in the results (in seconds). Must be provided if ``data`` has a times
        dimension.

        .. versionadded:: 1.3

    name : str  (default ``"CFC"``)
        Name of the results being stored.

    Methods
    -------
    get_results :
        Return the results.

    plot :
        Plot the results.

    Attributes
    ----------
    name : str
        Name of the results.

    indices : tuple of tuple of int, length of 2
        Indices of the channels for each connection of the results. Contains two tuples
        of equal length for the seed and target indices, respectively.

    shape : tuple of int
        Shape of the results i.e. ``[nodes, low frequencies, high frequencies]``.

    n_nodes : int
        Number of connections in the the results.

    f1s : ~numpy.ndarray, shape of [low frequencies]
        Low frequencies (in Hz) in the results.

    f2s : ~numpy.ndarray, shape of [high frequencies]
        High frequencies (in Hz) in the results.

    times : ~numpy.ndarray, shape of [times] | None
        Timepoints (in seconds) in the results.
    """

    def __repr__(self) -> str:
        """Return printable representation of the object."""
        repr_ = (
            f"<Result: {self.name} | [{self.n_nodes} nodes, {self.f1s.size} f1s, "
            f"{self.f2s.size} f2s"
        )
        if self.times is not None:
            repr_ += f", {self.times.size} timepoints"
        repr_ += "]>"

        return repr_

    def __init__(
        self,
        data: np.ndarray,
        indices: tuple[tuple[int]],
        f1s: np.ndarray,
        f2s: np.ndarray,
        times: np.ndarray | None = None,
        name: str = "CFC",
    ) -> None:  # noqa: D107
        super().__init__(data, (3, 4), name)
        self._sort_init_inputs(indices, f1s, f2s, times)

        self._plotting = _PlotCFC(
            data=self._data,
            indices=self.indices,
            f1s=self.f1s,
            f2s=self.f2s,
            times=self.times,
            name=self.name,
        )

    def _sort_init_inputs(
        self,
        indices: tuple[tuple[int]],
        f1s: np.ndarray,
        f2s: np.ndarray,
        times: np.ndarray | None,
    ) -> None:
        """Sort inputs to the object."""
        super()._sort_indices_seeds_targets(indices)
        super()._sort_freq_inputs(f1s, f2s)
        super()._sort_times(times)
        super()._check_data_shape()

    def _get_compact_results_child(self) -> tuple[np.ndarray, tuple[tuple[int]]]:
        """Return a compacted form of the results.

        Returns
        -------
        compact_results : numpy.ndarray of float
            Results with shape ``[seeds, targets, f1s, f2s]``.

        indices : tuple of tuple of int, length of 2
            Channel indices of ``compact_results`` for the seeds and targets,
            respectively.
        """
        compact_results = np.full(
            (self._n_chans, self._n_chans, self.f1s.shape[0], self.f2s.shape[0]),
            fill_value=np.full(
                (self.f1s.shape[0], self.f2s.shape[0]), fill_value=np.nan
            ),
        )

        return super()._get_compact_results_parent(compact_results)

    def plot(
        self,
        nodes: int | tuple[int] | None = None,
        f1s: tuple[int | float] | None = None,
        f2s: tuple[int | float] | None = None,
        times: tuple[int | float] | None = None,
        n_rows: int = 1,
        n_cols: int = 1,
        major_tick_intervals: int | float = 5.0,
        minor_tick_intervals: int | float = 1.0,
        cbar_range: tuple[float] | list[tuple[float]] | None = None,
        show: bool = True,
    ) -> tuple[list[Figure], list[np.ndarray]]:
        """Plot the results.

        Parameters
        ----------
        nodes : int | tuple of int | None (default None)
            Indices of connections to plot. If :obj:`None`, plot all connections.

        f1s : tuple of int or float | None (default None)
            Start and end low frequencies of the results to plot, respectively. If
            :obj:`None`, all low frequencies are plotted.

        f2s : tuple of int or float | None (default None)
            Start and end high frequencies of the results to plot, respectively. If
            :obj:`None`, all high frequencies are plotted.

        times : tuple of int or float, length of 2 | None (default None)
            Start and end times (in seconds) of the results to plot, respectively. If
            :obj:`None`, all timepoints are used.

            .. versionadded:: 1.3

        n_rows : int (default ``1``)
            Number of rows of subplots per figure.

        n_cols : int (default ``1``)
            Number of columns of subplots per figure.

        major_tick_intervals : int | float (default ``5.0``)
            Intervals (in Hz) at which the major ticks of the x- and y-axes should
            occur.

        minor_tick_intervals : int | float (default ``1.0``)
            Intervals (in Hz) at which the minor ticks of the x- and y-axes should
            occur.

        cbar_range : tuple of float | list of tuple of float | None (default None)
            Range (in units of the data) for the colourbars, consisting of the lower and
            upper limits, respectively. If :obj:`None`, the range is computed
            automatically. If a tuple of float, this range is used for all plots. If a
            list of tuple of float, the ranges are used for each individual plot.

        show : bool (default True)
            Whether or not to show the plotted results.

        Returns
        -------
        figures : list of matplotlib Figure
            Figures of the results in a list of length ``ceil(n_nodes / (n_rows *
            n_cols))``.

        axes : list of ~numpy.ndarray of matplotlib pyplot Axes
            Subplot axes for the results in a list of length ``ceil(n_nodes / (n_rows *
            n_cols))`` where each entry is a 1D ``~numpy.ndarray`` of length ``(n_rows *
            n_cols)``.

        Notes
        -----
        ``n_rows`` and ``n_cols`` of ``1`` will plot the results for each connection on
        a new figure.
        """
        figures, axes = self._plotting.plot(
            nodes=nodes,
            f1s=f1s,
            f2s=f2s,
            times=times,
            n_rows=n_rows,
            n_cols=n_cols,
            major_tick_intervals=major_tick_intervals,
            minor_tick_intervals=minor_tick_intervals,
            cbar_range=cbar_range,
            show=show,
        )

        return figures, axes


class ResultsTDE(_ResultsBase):
    """Class for storing time delay estimation (TDE) results.

    Parameters
    ----------
    data : ~numpy.ndarray, shape of [nodes, frequency_bands, times]
        Results to store.

    indices : tuple of tuple of int, length of 2
        Indices of the channels for each connection of the results. Should contain two
        tuples of equal length for the seed and target indices, respectively.

    times : ~numpy.ndarray, shape of [times]
        Timepoints in the results (in ms).

    freq_bands : tuple of tuple of int or float, length of 2 | None (default None)
        Lower and higher frequencies (in Hz), respectively, of each frequency band used
        to compute the results.

    name : str (default ``"TDE"``)
        Name of the results being stored.

    Methods
    -------
    get_results :
        Return the results.

    plot :
        Plot the results.

    Attributes
    ----------
    name : str
        Name of the results.

    indices : tuple of tuple of int, length of 2
        Indices of the channels for each connection in the results. Contains two tuples
        of equal length for the seed and target indices, respectively.

    shape : tuple of int
        Shape of the results i.e. ``[nodes, frequency bands, times]``.

    n_nodes : int
        Number of connections in the results.

    times : ~numpy.ndarray, shape of [times]
        Timepoints in the results (in ms).

    freq_bands : tuple of tuple of int or float, length of 2
        Lower and higher frequencies (in Hz), respectively, of each frequency band used
        to compute the results.

    tau : ~numpy.ndarray, shape of [nodes, frequency_bands]
        Estimated time delay (in ms) for each connection and frequency band.
    """  # noqa: E501

    freq_bands: tuple[tuple[int | float]] = None
    _n_fbands: int = None

    def __repr__(self) -> str:
        """Return printable representation of the object."""
        repr_ = f"<Result: {self.name} | "

        if self.freq_bands is not None:
            repr_ += (
                f"{np.min(self.freq_bands):.2f} - {np.max(self.freq_bands):.2f} Hz | "
            )

        repr_ += (
            f"[{self.n_nodes} nodes, {self._n_fbands} frequency bands, "
            f"{self.times.size} timepoints]>"
        )

        return repr_

    def __init__(
        self,
        data: np.ndarray,
        indices: tuple[tuple[int]],
        times: np.ndarray,
        freq_bands: tuple[tuple[int | float]] | None = None,
        name: str = "TDE",
    ) -> None:  # noqa: D107
        super().__init__(data, (3,), name)
        self._sort_init_inputs(indices, times, freq_bands)

        self._compute_tau()

        self._plotting = _PlotTDE(
            data=self._data,
            tau=self.tau,
            indices=self.indices,
            freq_bands=self.freq_bands,
            times=self.times,
            name=self.name,
        )

    def _sort_init_inputs(
        self,
        indices: tuple[tuple[int]],
        times: np.ndarray,
        freq_bands: tuple[int | float],
    ) -> None:
        """Sort inputs to the object."""
        super()._sort_indices_seeds_targets(indices)
        self._sort_times(times)
        self._sort_freq_bands(freq_bands)

        if self._data.shape != (self.n_nodes, self._n_fbands, times.shape[0]):
            raise ValueError("`data` must have shape [nodes, frequency bands, times].")

    def _sort_times(self, times: np.ndarray) -> None:
        """Sort ``times`` input."""
        if not isinstance(times, np.ndarray):
            raise TypeError("`times` must be a NumPy array.")
        if times.ndim != 1:
            raise ValueError("`times` must be a 1D array.")

        self.times = times

    def _sort_freq_bands(self, freq_bands: tuple[tuple[int | float]]) -> None:
        """Sort ``freq_bands`` input."""
        if freq_bands is not None:
            if not isinstance(freq_bands, tuple):
                raise TypeError("`freq_bands` must be a tuple.")
            if len(freq_bands) != self._data.shape[1]:
                raise ValueError(
                    "`freq_bands` must the same length as the number of frequency "
                    "bands in the results."
                )
            for freq_band in freq_bands:
                if not isinstance(freq_band, tuple):
                    raise TypeError("Each entry of `freq_bands` must be a tuple.")
                if len(freq_band) != 2:
                    raise ValueError(
                        "Each entry of `freq_bands` must have length of 2."
                    )

            self.freq_bands = freq_bands
            self._n_fbands = len(freq_bands)
        else:
            self._n_fbands = self._data.shape[1]

    def _get_compact_results_child(self) -> tuple[np.ndarray, tuple[np.ndarray]]:
        """Return a compacted form of the results.

        Returns
        -------
        compact_results : numpy.ndarray
            Results with shape ``[seeds, targets, frequency bands, times]``.

        indices : tuple of tuple of int, length of 2
            Channel indices of ``compact_results`` for the seeds and targets,
            respectively.
        """
        compact_results = np.full(
            (self._n_chans, self._n_chans, self._n_fbands, self.times.shape[0]),
            fill_value=np.full(
                (self._n_fbands, self.times.shape[0]), fill_value=np.nan
            ),
        )

        return super()._get_compact_results_parent(compact_results)

    def plot(
        self,
        nodes: int | tuple[int] | None = None,
        freq_bands: int | tuple[int] | None = None,
        times: tuple[int | float] | None = None,
        n_rows: int = 1,
        n_cols: int = 1,
        major_tick_intervals: int | float = 500.0,
        minor_tick_intervals: int | float = 100.0,
        show: bool = True,
    ) -> tuple[list[Figure], list[np.ndarray]]:
        """Plot the results.

        Parameters
        ----------
        nodes : int | tuple of int | None (default None)
            Indices of connections to plot. If :obj:`None`, all connections are plotted.

        freq_bands : int | tuple of int | None (default None)
            Indices of frequency bands to plot. If :obj:`None`, all frequency bands are
            plotted.

        times : tuple of int or float | None (default None)
            Start and end times of the results to plot. If :obj:`None`, plot all times.

        n_rows : int (default ``1``)
            Number of rows of subplots per figure.

        n_cols : int (default ``1``)
            Number of columns of subplots per figure.

        major_tick_intervals : int | float (default ``500.0``)
            Intervals (in ms) at which the major ticks of the x- and y-axes should
            occur.

        minor_tick_intervals : int | float (default ``100.0``)
            Intervals (in ms) at which the minor ticks of the x- and y-axes should
            occur.

        show : bool (default True)
            Whether or not to show the plotted results.

        Returns
        -------
        figures : list of matplotlib Figure
            Figures of the results in a list of length ``ceil(n_nodes / (n_rows *
            n_cols))``.

        axes : list of ~numpy.ndarray of matplotlib pyplot Axes
            Subplot axes for the results in a list of length ``ceil(n_nodes / (n_rows *
            n_cols))`` where each entry is a 1D :obj:`~numpy.ndarray` of length
            ``(n_rows * n_cols)``.

        Notes
        -----
        ``n_rows`` and ``n_cols`` of ``1`` will plot the results for each connection on
        a new figure.
        """
        figures, axes = self._plotting.plot(
            nodes=nodes,
            freq_bands=freq_bands,
            times=times,
            n_rows=n_rows,
            n_cols=n_cols,
            major_tick_intervals=major_tick_intervals,
            minor_tick_intervals=minor_tick_intervals,
            show=show,
        )

        return figures, axes

    def _compute_tau(self) -> None:
        """Compute the time delay estimates for each connection."""
        self._tau = []
        for node_i in range(self.n_nodes):
            self._tau.append(self.times[self._data[node_i].argmax(axis=1)])
        self._tau = np.array(self._tau)

    @property
    def tau(self) -> np.ndarray:
        return self._tau.copy()


class ResultsWaveShape(_ResultsBase):
    """Class for storing wave shape results.

    Parameters
    ----------
    data : ~numpy.ndarray, shape of [nodes, low frequencies, high frequencies (, times)]
        Results to store.

    indices : tuple of int
        Indices of the channels in the results.

    f1s : ~numpy.ndarray, shape of [low frequencies]
        Low frequencies (in Hz) in the results.

    f2s : ~numpy.ndarray, shape of [high frequencies]
        High frequencies (in Hz) in the results.

    times : ~numpy.ndarray, shape of [times] | None (default None)
        Timepoints in the results (in seconds). Must be provided if ``data`` has a times
        dimension.

        .. versionadded:: 1.3

    name : str (default ``"Waveshape"``)
        Name of the results being stored.

    Methods
    -------
    get_results :
        Return the results.

    plot :
        Plot the results.

    Attributes
    ----------
    name : str
        Name of the results.

    indices : tuple of int
        Indices of the channels in the results.

    shape : tuple of int
        Shape of the results i.e. ``[nodes, low frequencies, high frequencies]``.

    n_nodes : int
        Number of channels in the results.

    f1s : ~numpy.ndarray, shape of [low frequencies]
        Low frequencies (in Hz) in the results.

    f2s : ~numpy.ndarray, shape of [high frequencies]
        High frequencies (in Hz) in the results.

    times : ~numpy.ndarray, shape of [times] | None
        Timepoints (in seconds) in the results.
    """

    def __repr__(self) -> str:
        """Return printable representation of the object."""
        repr_ = (
            f"<Result: {self.name} | [{self.n_nodes} nodes, {self.f1s.size} f1s, "
            f"{self.f2s.size} f2s"
        )
        if self.times is not None:
            repr_ += f", {self.times.size} timepoints"
        repr_ += "]>"

        return repr_

    def __init__(
        self,
        data: np.ndarray,
        indices: tuple[int],
        f1s: np.ndarray,
        f2s: np.ndarray,
        times: np.ndarray | None = None,
        name: str = "Waveshape",
    ) -> None:  # noqa: D107
        super().__init__(data, (3, 4), name)
        self._sort_init_inputs(indices, f1s, f2s, times)

        self._plotting = _PlotWaveShape(
            data=self._data,
            indices=self.indices,
            f1s=self.f1s,
            f2s=self.f2s,
            times=self.times,
            name=self.name,
        )

    def _sort_init_inputs(
        self,
        indices: tuple[int],
        f1s: np.ndarray,
        f2s: np.ndarray,
        times: np.ndarray | None,
    ) -> None:
        """Sort inputs to the object."""
        super()._sort_indices_channels(indices)
        super()._sort_freq_inputs(f1s, f2s)
        super()._sort_times(times)
        super()._check_data_shape()

    def get_results(self, copy: bool = True) -> np.ndarray:
        """Return the results.

        Parameters
        ----------
        copy : bool (default True)
            Whether or not to return a copy of the results.

            .. versionadded:: 1.2

        Returns
        -------
        results : ~numpy.ndarray, shape of [nodes, low frequencies, high frequencies]
            The results.
        """
        if not isinstance(copy, bool):
            raise TypeError("`copy` must be a bool.")

        if copy:
            return self._data.copy()
        return self._data

    def plot(
        self,
        nodes: int | tuple[int] | None = None,
        f1s: tuple[int | float] | None = None,
        f2s: tuple[int | float] | None = None,
        times: tuple[int | float] | None = None,
        n_rows: int = 1,
        n_cols: int = 1,
        major_tick_intervals: int | float = 5.0,
        minor_tick_intervals: int | float = 1.0,
        plot_absolute: bool = False,
        mirror_cbar_range: bool = True,
        cbar_range_abs: tuple[float] | list[tuple[float]] | None = None,
        cbar_range_real: tuple[float] | list[tuple[float]] | None = None,
        cbar_range_imag: tuple[float] | list[tuple[float]] | None = None,
        cbar_range_phase: tuple[float] | list[tuple[float]] | None = None,
        show: bool = True,
    ) -> tuple[list[Figure], list[np.ndarray]]:
        """Plot the results.

        Parameters
        ----------
        nodes : int | tuple of int | None (default None)
            Indices of results of channels to plot. If :obj:`None`, plot results of all
            channels.

        f1s : tuple of int or float | None (default None)
            Start and end low frequencies of the results to plot, respectively. If
            :obj:`None`, plot all low frequencies.

        f2s : tuple of int or float | None (default None)
            Start and end high frequencies of the results to plot, respectively. If
            :obj:`None`, plot all high frequencies.

        times : tuple of int or float, length of 2 | None (default None)
            Start and end times (in seconds) of the results to plot, respectively. If
            :obj:`None`, all timepoints are used.

            .. versionadded:: 1.3

        n_rows : int (default ``1``)
            Number of rows of subplots per figure.

        n_cols : int (default ``1``)
            Number of columns of subplots per figure.

        major_tick_intervals : int | float (default ``5.0``)
            Intervals (in Hz) at which the major ticks of the x- and y-axes should
            occur.

        minor_tick_intervals : int | float (default ``1.0``)
            Intervals (in Hz) at which the minor ticks of the x- and y-axes should
            occur.

        plot_absolute : bool (default False)
            Whether or not to plot the absolute values of the real and imaginary parts
            of the results.

        mirror_cbar_range : bool (default True)
            Whether of not to mirror the colourbar ranges of the real and imaginary
            results around 0. Only applied if ``plot_absolute`` is :obj:`False`, and
            ``cbar_range_real`` and ``cbar_range_imag`` are not :obj:`None`.

        cbar_range_abs : tuple of float | list of tuple of float | None (default None)
            Range (in units of the data) for the colourbars of the absolute value of the
            results, consisting of the lower and upper limits, respectively. If
            :obj:`None`, the range is computed automatically. If a tuple of float, this
            range is used for all plots. If a tuple of tuple of float, the ranges are
            used for each individual plot.

        cbar_range_real : tuple of float | list of tuple of float | None (default None)
            Range (in units of the data) for the colourbars of the real value of the
            results, consisting of the lower and upper limits, respectively. If
            :obj:`None`, the range is computed automatically. If a tuple of float, this
            range is used for all plots. If a tuple of tuple of float, the ranges are
            used for each individual plot.

        cbar_range_imag : tuple of float | list of tuple of float | None (default None)
            Range (in units of the data) for the colourbars of the imaginary value of
            the results, consisting of the lower and upper limits, respectively. If
            :obj:`None`, the range is computed automatically. If a tuple of float, this
            range is used for all plots. If a tuple of tuple of float, the ranges are
            used for each individual plot.

        cbar_range_phase : tuple of float | list of tuple of float | None (default None)
            Range (in units of the data) for the colourbars of the phase of the results,
            consisting of the lower and upper limits, respectively. If :obj:`None`, the
            range is computed automatically. If a tuple of float, this range is used for
            all plots. If a tuple of tuple of float, the ranges are used for each
            individual plot. Note that results are limited to the range (-pi, pi].

        show : bool (default True)
            Whether or not to show the plotted results.

        Returns
        -------
        figures : list of matplotlib Figure
            Figures of the results in a list of length ``ceil(n_nodes / (n_rows *
            n_cols))``.

        axes : list of ~numpy.ndarray of ~numpy.ndarray of matplotlib pyplot Axes
            Subplot axes for the results in a list of length ``ceil(n_nodes / (n_rows *
            n_cols))`` where each entry is a 1D :obj:`~numpy.ndarray` of length
            ``(n_rows * n_cols)``, whose entries are themselves 1D :obj:`~numpy.ndarray`
            of length 4, corresponding to the absolute, real, imaginary, and phase
            plots, respectively.

        Notes
        -----
        ``n_rows`` and ``n_cols`` of ``1`` will plot the results for each node on a new
        figure.
        """
        figures, axes = self._plotting.plot(
            nodes=nodes,
            f1s=f1s,
            f2s=f2s,
            times=times,
            n_rows=n_rows,
            n_cols=n_cols,
            major_tick_intervals=major_tick_intervals,
            minor_tick_intervals=minor_tick_intervals,
            plot_absolute=plot_absolute,
            mirror_cbar_range=mirror_cbar_range,
            cbar_range_abs=cbar_range_abs,
            cbar_range_real=cbar_range_real,
            cbar_range_imag=cbar_range_imag,
            cbar_range_phase=cbar_range_phase,
            show=show,
        )

        return figures, axes


class ResultsGeneral(_ResultsBase):
    """Class for storing general bispectrum and threenorm results.

    Parameters
    ----------
    data : ~numpy.ndarray, shape of [nodes, low frequencies, high frequencies (, times)]
        Results to store.

    indices : tuple of tuple of int, length of 3
        Indices of the channels for each combination of the results. Should contain
        three tuples of equal length for the k, m, and n channel indices, respectively.

    f1s : ~numpy.ndarray, shape of [low frequencies]
        Low frequencies (in Hz) in the results.

    f2s : ~numpy.ndarray, shape of [high frequencies]
        High frequencies (in Hz) in the results.

    times : ~numpy.ndarray, shape of [times] | None (default None)
        Timepoints in the results (in seconds). Must be provided if ``data`` has a times
        dimension.

        .. versionadded:: 1.3

    name : str  (default ``"General"``)
        Name of the results being stored.

    Methods
    -------
    get_results :
        Return the results.

    plot :
        Plot the results.

    Attributes
    ----------
    name : str
        Name of the results.

    indices : tuple of tuple of int, length of 2
        Indices of the channels for each connection of the results. Contains three
        tuples of equal length for the k, m, and n channel indices, respectively.

    shape : tuple of int
        Shape of the results i.e. ``[nodes, low frequencies, high frequencies]``.

    n_nodes : int
        Number of connections in the the results.

    f1s : ~numpy.ndarray, shape of [low frequencies]
        Low frequencies (in Hz) in the results.

    f2s : ~numpy.ndarray, shape of [high frequencies]
        High frequencies (in Hz) in the results.

    times : ~numpy.ndarray, shape of [times] | None
        Timepoints (in seconds) in the results.

    Notes
    -----

    .. versionadded:: 1.2
    """

    def __repr__(self) -> str:
        """Return printable representation of the object."""
        repr_ = (
            f"<Result: {self.name} | [{self.n_nodes} nodes, {self.f1s.size} f1s, "
            f"{self.f2s.size} f2s"
        )
        if self.times is not None:
            repr_ += f", {self.times.size} timepoints"
        repr_ += "]>"

        return repr_

    def __init__(
        self,
        data: np.ndarray,
        indices: tuple[tuple[int]],
        f1s: np.ndarray,
        f2s: np.ndarray,
        times: np.ndarray | None = None,
        name: str = "General",
    ) -> None:  # noqa: D107
        super().__init__(data, (3, 4), name)
        self._sort_init_inputs(indices, f1s, f2s, times)

        self._plotting = _PlotGeneral(
            data=self._data,
            indices=self.indices,
            f1s=self.f1s,
            f2s=self.f2s,
            times=self.times,
            name=self.name,
        )

    def _sort_init_inputs(
        self,
        indices: tuple[tuple[int]],
        f1s: np.ndarray,
        f2s: np.ndarray,
        times: np.ndarray | None,
    ) -> None:
        """Sort inputs to the object."""
        super()._sort_indices_kmn(indices)
        super()._sort_freq_inputs(f1s, f2s)
        super()._sort_times(times)
        super()._check_data_shape()

    def get_results(
        self, form: str = "raveled", copy=True
    ) -> np.ndarray | tuple[np.ndarray, tuple[tuple[int]]]:
        """Return the results.

        Parameters
        ----------
        form : ``"raveled"`` | ``"compact"`` (default ``"raveled"``)
            How the results should be returned: ``"raveled"`` - results have shape
            ``[nodes, ...]``; ``"compact"`` - results have shape ``[k, m, n, ...]``,
            where ``...`` represents the data dimensions (e.g. frequencies, times).

        copy : bool (default True)
            Whether to return a copy of the results.

            .. versionadded:: 1.2

        Returns
        -------
        results : ~numpy.ndarray
            The results.

        indices : tuple of tuple of int, length of 3
            Channel indices of the k, m, and n channels in ``results``, according to the
            node order in the original data indices. Only returned if ``form`` is
            ``"compact"``.
        """
        return super().get_results(form, copy)

    def _get_compact_results_child(self) -> tuple[np.ndarray, tuple[tuple[int]]]:
        """Return a compacted form of the results.

        Returns
        -------
        compact_results : numpy.ndarray of float
            Results with shape ``[k, m, n, f1s, f2s]``.

        indices : tuple of tuple of int, length of 3
            Channel indices of ``compact_results`` for the k, m, and n channels,
            respectively.
        """
        compact_results = np.full(
            (
                self._n_chans,
                self._n_chans,
                self._n_chans,
                self.f1s.shape[0],
                self.f2s.shape[0],
            ),
            fill_value=np.full(
                (self.f1s.shape[0], self.f2s.shape[0]), fill_value=np.nan
            ),
        )

        for con_result, k, m, n in zip(
            self._data, self._kmn[0], self._kmn[1], self._kmn[2]
        ):
            compact_results[k, m, n] = con_result

        indices = tuple(group_idcs for group_idcs in self._kmn)

        return compact_results, indices

    def plot(
        self,
        nodes: int | tuple[int] | None = None,
        f1s: tuple[int | float] | None = None,
        f2s: tuple[int | float] | None = None,
        times: tuple[int | float] | None = None,
        n_rows: int = 1,
        n_cols: int = 1,
        major_tick_intervals: int | float = 5.0,
        minor_tick_intervals: int | float = 1.0,
        plot_absolute: bool = False,
        mirror_cbar_range: bool = True,
        cbar_range_abs: tuple[float] | list[tuple[float]] | None = None,
        cbar_range_real: tuple[float] | list[tuple[float]] | None = None,
        cbar_range_imag: tuple[float] | list[tuple[float]] | None = None,
        cbar_range_phase: tuple[float] | list[tuple[float]] | None = None,
        show: bool = True,
    ) -> tuple[list[Figure], list[np.ndarray]]:
        """Plot the results.

        Parameters
        ----------
        nodes : int | tuple of int | None (default None)
            Indices of results of channels to plot. If :obj:`None`, plot results of all
            channels.

        f1s : tuple of int or float | None (default None)
            Start and end low frequencies of the results to plot, respectively. If
            :obj:`None`, plot all low frequencies.

        f2s : tuple of int or float | None (default None)
            Start and end high frequencies of the results to plot, respectively. If
            :obj:`None`, plot all high frequencies.

        times : tuple of int or float, length of 2 | None (default None)
            Start and end times (in seconds) of the results to plot, respectively. If
            :obj:`None`, all timepoints are used.

            .. versionadded:: 1.3

        n_rows : int (default ``1``)
            Number of rows of subplots per figure.

        n_cols : int (default ``1``)
            Number of columns of subplots per figure.

        major_tick_intervals : int | float (default ``5.0``)
            Intervals (in Hz) at which the major ticks of the x- and y-axes should
            occur.

        minor_tick_intervals : int | float (default ``1.0``)
            Intervals (in Hz) at which the minor ticks of the x- and y-axes should
            occur.

        plot_absolute : bool (default False)
            Whether or not to plot the absolute values of the real and imaginary parts
            of the results.

        mirror_cbar_range : bool (default True)
            Whether of not to mirror the colourbar ranges of the real and imaginary
            results around 0. Only applied if ``plot_absolute`` is :obj:`False`, and
            ``cbar_range_real`` and ``cbar_range_imag`` are not :obj:`None`.

        cbar_range_abs : tuple of float | list of tuple of float | None (default None)
            Range (in units of the data) for the colourbars of the absolute value of the
            results, consisting of the lower and upper limits, respectively. If
            :obj:`None`, the range is computed automatically. If a tuple of float, this
            range is used for all plots. If a tuple of tuple of float, the ranges are
            used for each individual plot.

        cbar_range_real : tuple of float | list of tuple of float | None (default None)
            Range (in units of the data) for the colourbars of the real value of the
            results, consisting of the lower and upper limits, respectively. If
            :obj:`None`, the range is computed automatically. If a tuple of float, this
            range is used for all plots. If a tuple of tuple of float, the ranges are
            used for each individual plot.

        cbar_range_imag : tuple of float | list of tuple of float | None (default None)
            Range (in units of the data) for the colourbars of the imaginary value of
            the results, consisting of the lower and upper limits, respectively. If
            :obj:`None`, the range is computed automatically. If a tuple of float, this
            range is used for all plots. If a tuple of tuple of float, the ranges are
            used for each individual plot.

        cbar_range_phase : tuple of float | list of tuple of float | None (default None)
            Range (in units of the data) for the colourbars of the phase of the results,
            consisting of the lower and upper limits, respectively. If :obj:`None`, the
            range is computed automatically. If a tuple of float, this range is used for
            all plots. If a tuple of tuple of float, the ranges are used for each
            individual plot. Note that results are limited to the range (-pi, pi].

        show : bool (default True)
            Whether or not to show the plotted results.

        Returns
        -------
        figures : list of matplotlib Figure
            Figures of the results in a list of length ``ceil(n_nodes / (n_rows *
            n_cols))``.

        axes : list of ~numpy.ndarray of ~numpy.ndarray of matplotlib pyplot Axes
            Subplot axes for the results in a list of length ``ceil(n_nodes / (n_rows *
            n_cols))`` where each entry is a 1D :obj:`~numpy.ndarray` of length
            ``(n_rows * n_cols)``, whose entries are themselves 1D
            :obj:`~numpy.ndarray` of length 4, corresponding to the absolute, real,
            imaginary, and phase plots, respectively.

        Notes
        -----
        ``n_rows`` and ``n_cols`` of ``1`` will plot the results for each node on a new
        figure.
        """  # noqa: E501
        figures, axes = self._plotting.plot(
            nodes=nodes,
            f1s=f1s,
            f2s=f2s,
            times=times,
            n_rows=n_rows,
            n_cols=n_cols,
            major_tick_intervals=major_tick_intervals,
            minor_tick_intervals=minor_tick_intervals,
            plot_absolute=plot_absolute,
            mirror_cbar_range=mirror_cbar_range,
            cbar_range_abs=cbar_range_abs,
            cbar_range_real=cbar_range_real,
            cbar_range_imag=cbar_range_imag,
            cbar_range_phase=cbar_range_phase,
            show=show,
        )

        return figures, axes
