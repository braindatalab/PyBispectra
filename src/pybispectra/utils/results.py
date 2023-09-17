"""Helper tools for storing results."""

from abc import ABC, abstractmethod
from copy import deepcopy

from matplotlib.figure import Figure
import numpy as np

from pybispectra.utils._plot import _PlotCFC, _PlotTDE, _PlotWaveShape


class _ResultsBase(ABC):
    """Base class for storing results."""

    def __init__(
        self,
        data: np.ndarray,
        data_ndim: int,
        name: str,
    ) -> None:
        if not isinstance(data, np.ndarray):
            raise TypeError("`data` must be a NumPy array.")
        if data.ndim != data_ndim:
            raise ValueError(f"`data` must be a {data_ndim}D array.")
        self._data = data.copy()
        self.shape = data.shape

        if not isinstance(name, str):
            raise TypeError("`name` must be a string.")
        self.name = deepcopy(name)

    @abstractmethod
    def _sort_init_inputs(self) -> None:
        """Sort inputs to the object."""

    def _sort_freq_inputs(self, f1s: np.ndarray, f2s: np.ndarray) -> None:
        """Sort inputs to the object."""
        if not isinstance(f1s, np.ndarray) or not isinstance(f2s, np.ndarray):
            raise TypeError("`f1s` and `f2s` must be NumPy arrays.")
        if f1s.ndim != 1 or f2s.ndim != 1:
            raise ValueError("`f1s` and `f2s` must be 1D arrays.")
        self.f1s = f1s.copy()
        self.f2s = f2s.copy()

        if self._data.shape != (self.n_nodes, len(f1s), len(f2s)):
            raise ValueError("`data` must have shape [nodes, f1s, f2s].")

    def _sort_indices_seeds_targets(self, indices: tuple[tuple[int]]) -> None:
        """Sort `indices` inputs with format ([seeds], [targets])."""
        if not isinstance(indices, tuple):
            raise TypeError("`indices` must be a tuple.")
        if len(indices) != 2:
            raise ValueError("`indices` must have a length of 2.")

        seeds = indices[0]
        targets = indices[1]
        for group_idcs in (seeds, targets):
            if not isinstance(group_idcs, tuple):
                raise TypeError("Entries of `indices` must be tuples.")
            if any(
                not isinstance(idx, (int, np.integer)) for idx in group_idcs
            ):
                raise TypeError(
                    "Entries for seeds and targets in `indices` must be "
                    "ints."
                )
        if len(seeds) != len(targets):
            raise ValueError("Entries of `indices` must have equal length.")
        self._n_chans = len(np.unique([*seeds, *targets]))
        for group_idcs in (seeds, targets):
            if any(idx < 0 or idx >= self._n_chans for idx in group_idcs):
                raise ValueError(
                    "`indices` contains indices for nodes not present in "
                    "the data."
                )
        self._seeds = deepcopy(seeds)
        self._targets = deepcopy(targets)
        self.n_nodes = len(seeds)
        self.indices = deepcopy(indices)

    def _sort_indices_channels(self, indices: tuple[int]) -> None:
        """Sort `indices` with inputs format [channels]."""
        if not isinstance(indices, tuple):
            raise TypeError("`indices` must be a tuple.")
        if not all(isinstance(idx, (int, np.integer)) for idx in indices):
            raise TypeError("Entries of `indices` must be ints.")
        self._n_chans = len(np.unique(indices))
        if any(idx < 0 or idx >= self._n_chans for idx in indices):
            raise ValueError(
                "`indices` contains indices for channels not present in "
                "the data."
            )
        self.n_nodes = len(indices)
        self.indices = deepcopy(indices)

    def get_results(
        self, form: str = "raveled"
    ) -> np.ndarray | tuple[np.ndarray, tuple[tuple[int]]]:
        """Return a copy of the results.

        Parameters
        ----------
        form : str (default ``"raveled"``)
            How the results should be returned: ``"raveled"`` - results have
            shape `[nodes, ...]`; ``"compact"`` - results have shape
            ``[seeds, targets, ...]``, where ``...`` represents the data
            dimensions (e.g. frequencies, times).

        Returns
        -------
        results : ~numpy.ndarray
            The results.

        indices : tuple of tuple of int, length of 2
            Channel indices of the seeds and targets. Only returned if ``form``
            is ``"compact"``.
        """
        accepted_forms = ["raveled", "compact"]
        if form not in accepted_forms:
            raise ValueError("`form` is not recognised.")

        if form == "raveled":
            return self._data.copy()
        return self._get_compact_results_child()

    def _get_compact_results_child(self) -> None:
        """Return a compacted form of the results."""

    def _get_compact_results_parent(
        self, compact_results: np.ndarray
    ) -> tuple[np.ndarray, tuple[tuple[int]]]:
        """Return a compacted form of the results.

        Parameters
        ----------
        compact_results : numpy.ndarray
            Empty results array with shape ``[seeds, targets, ...]``, where
            ``...`` represents the data dimensions (e.g. frequencies, times).

        Returns
        -------
        compact_results : numpy.ndarray
            Results array with shape ``[seeds, targets, ...]``, where ``...``
            represents the data dimensions (e.g. frequencies, times).

        indices : tuple of tuple of int
            Channel indices of ``compact_results`` for the seeds and targets,
            respectively.
        """
        for con_result, seed, target in zip(
            self._data, self._seeds, self._targets
        ):
            compact_results[seed, target] = con_result

        # remove empty rows and cols
        filled_rows = []
        for row_i, row in enumerate(compact_results):
            if not all(np.isnan(entry).all() for entry in row):
                filled_rows.append(row_i)
        filled_cols = []
        for col_i, col in enumerate(compact_results.swapaxes(1, 0)):
            if not all(np.isnan(entry).all() for entry in col):
                filled_cols.append(col_i)
        compact_results = compact_results[np.ix_(filled_rows, filled_cols)]

        indices = (
            tuple(np.unique(self._seeds).tolist()),
            tuple(np.unique(self._targets).tolist()),
        )

        return compact_results.copy(), indices


class ResultsCFC(_ResultsBase):
    """Class for storing cross-frequency coupling (CFC) results.

    Parameters
    ----------
    data : ~numpy.ndarray, shape of [nodes, f1s, f2s]
        Results to store.

    indices : tuple of tuple of int, length of 2
        Indices of the channels for each connection of the results. Should
        contain two tuples of equal length for the seed and target indices,
        respectively.

    f1s : ~numpy.ndarray, shape of [low frequencies]
        Low frequencies (in Hz) in the results.

    f2s : ~numpy.ndarray, shape of [high frequencies]
        High frequencies (in Hz) in the results.

    name : str
        Name of the results being stored.

    Methods
    -------
    plot :
        Plot the results.

    get_results :
        Return a copy of the results.

    Attributes
    ----------
    name : str
        Name of the results.

    indices : tuple of tuple of int, length of 2
        Indices of the channels for each connection of the results. Should
        contain two tuples of equal length for the seed and target indices,
        respectively.

    shape : tuple of int
        Shape of the results i.e. [nodes, f1s, f2s].

    n_nodes : int
        Number of connections in the the results.

    f1s : ~numpy.ndarray, shape of [low frequencies]
        Low frequencies (in Hz) in the results.

    f2s : ~numpy.ndarray, shape of [high frequencies]
        High frequencies (in Hz) in the results.
    """

    def __repr__(self) -> str:
        """Return printable representation of the object."""
        return repr(
            f"<Result: {self.name} | [{self.n_nodes} nodes, "
            f"{len(self.f1s)} f1s, {len(self.f2s)} f2s]>"
        )

    def __init__(
        self,
        data: np.ndarray,
        indices: tuple[tuple[int]],
        f1s: np.ndarray,
        f2s: np.ndarray,
        name: str,
    ) -> None:  # noqa D107
        super().__init__(data, 3, name)
        self._sort_init_inputs(indices, f1s, f2s)

        self._plotting = _PlotCFC(
            data=self._data,
            indices=self.indices,
            f1s=self.f1s,
            f2s=self.f2s,
            name=self.name,
        )

    def _sort_init_inputs(
        self,
        indices: tuple[tuple[int]],
        f1s: np.ndarray,
        f2s: np.ndarray,
    ) -> None:
        """Sort inputs to the object."""
        super()._sort_indices_seeds_targets(indices)
        super()._sort_freq_inputs(f1s, f2s)

    def _get_compact_results_child(
        self,
    ) -> tuple[np.ndarray, tuple[tuple[int]]]:
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
            (
                self._n_chans,
                self._n_chans,
                self.f1s.shape[0],
                self.f2s.shape[0],
            ),
            fill_value=np.full(
                (self.f1s.shape[0], self.f2s.shape[0]), fill_value=np.nan
            ),
        )

        return super()._get_compact_results_parent(compact_results)

    def plot(
        self,
        nodes: tuple[int] | None = None,
        f1s: tuple[int | float] | None = None,
        f2s: tuple[int | float] | None = None,
        n_rows: int = 1,
        n_cols: int = 1,
        major_tick_intervals: int | float = 5.0,
        minor_tick_intervals: int | float = 1.0,
        cbar_range: tuple[float] | tuple[tuple[float]] | None = None,
        show: bool = True,
    ) -> tuple[list[Figure], list[np.ndarray]]:
        """Plot the results.

        Parameters
        ----------
        nodes : tuple of int | None (default None)
            Indices of connections to plot. If :obj:`None`, plot all
            connections.

        f1s : tuple of int or float | None (default None)
            Start and end low frequencies of the results to plot, respectively.
            If :obj:`None`, all low frequencies are plotted.

        f2s : tuple of int or float | None (default None)
            Start and end high frequencies of the results to plot,
            respectively. If :obj:`None`, all high frequencies are plotted.

        n_rows : int (default ``1``)
            Number of rows of subplots per figure.

        n_cols : int (default ``1``)
            Number of columns of subplots per figure.

        major_tick_intervals : int | float (default ``5.0``)
            Intervals (in Hz) at which the major ticks of the x- and y-axes
            should occur.

        minor_tick_intervals : int | float (default ``1.0``)
            Intervals (in Hz) at which the minor ticks of the x- and y-axes
            should occur.

        cbar_range : tuple of float | tuple of tuple of float | None (default None)
            Range (in units of the data) for the colourbars, consisting of the
            lower and upper limits, respectively. If :obj:`None`, the range is
            computed automatically. If a tuple of float, this range is used for
            all plots. If a tuple of tuple of float, the ranges are used for
            each individual plot.

        show : bool (default ``True``)
            Whether or not to show the plotted results.

        Returns
        -------
        figures : list of matplotlib Figure
            Figures of the results in a list of length
            ``ceil(n_nodes / (n_rows * n_cols))``.

        axes : list of ~numpy.ndarray of matplotlib pyplot Axes
            Subplot axes for the results in a list of length
            ``ceil(n_nodes / (n_rows * n_cols))`` where each entry is a 1D
            ``numpy.ndarray`` of length ``(n_rows * n_cols)``.

        Notes
        -----
        ``n_rows`` and ``n_cols`` of ``1`` will plot the results for each
        connection on a new figure.
        """  # noqa: E501
        figures, axes = self._plotting.plot(
            nodes=nodes,
            f1s=f1s,
            f2s=f2s,
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
    data : ~numpy.ndarray, shape of [nodes, times]
        Results to store.

    indices : tuple of tuple of int, length of 2
        Indices of the channels for each connection of the results. Should
        contain two tuples of equal length for the seed and target indices,
        respectively.

    times : ~numpy.ndarray, shape of [times]
        Timepoints in the results (in ms).

    name : str
        Name of the results being stored.

    Methods
    -------
    plot :
        Plot the results.

    get_results :
        Return a copy of the results.

    Attributes
    ----------
    name : str
        Name of the results.

    indices : tuple of tuple of int, length of 2
        Indices of the channels for each connection in the results. Contains
        two tuples of equal length for the seed and target indices,
        respectively.

    shape : tuple of int
        Shape of the results i.e. [nodes, times].

    n_nodes : str
        Number of connections in the results.

    times : ~numpy.ndarray, shape of [times]
        Timepoints in the results (in ms).

    tau : tuple of float
        Estimated time delay for each connection (in ms).
    """

    def __repr__(self) -> str:
        """Return printable representation of the object."""
        return repr(
            f"<Result: {self.name} | [{self.n_nodes} nodes, "
            f"{len(self.times)} times]>"
        )

    def __init__(
        self,
        data: np.ndarray,
        indices: tuple[tuple[int]],
        times: np.ndarray,
        name: str,
    ) -> None:  # noqa D107
        super().__init__(data, 2, name)
        self._sort_init_inputs(indices, times)

        self._compute_tau()

        self._plotting = _PlotTDE(
            data=self._data,
            tau=self.tau,
            indices=self.indices,
            times=self.times,
            name=self.name,
        )

    def _sort_init_inputs(
        self, indices: tuple[tuple[int]], times: np.ndarray
    ) -> None:
        """Sort inputs to the object."""
        super()._sort_indices_seeds_targets(indices)
        self._sort_times(times)

    def _sort_times(self, times: np.ndarray) -> None:
        """Sort `times` input."""
        if not isinstance(times, np.ndarray):
            raise TypeError("`times` must be a NumPy array.")
        if times.ndim != 1:
            raise ValueError("`times` must be a 1D array.")

        if self._data.shape != (self.n_nodes, times.shape[0]):
            raise ValueError("`data` must have shape [nodes, times].")

        self.times = times.copy()

    def _get_compact_results_child(
        self,
    ) -> tuple[np.ndarray, tuple[np.ndarray]]:
        """Return a compacted form of the results.

        Returns
        -------
        compact_results : numpy.ndarray
            Results with shape ``[seeds, targets, times]``.

        indices : tuple of tuple of int, length of 2
            Channel indices of ``compact_results`` for the seeds and targets,
            respectively.
        """
        compact_results = np.full(
            (self._n_chans, self._n_chans, self.times.shape[0]),
            fill_value=np.full(self.times.shape[0], fill_value=np.nan),
        )

        return super()._get_compact_results_parent(compact_results)

    def plot(
        self,
        nodes: tuple[int] | None = None,
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
        nodes : tuple of int | None (default None)
            Indices of connections to plot. If :obj:`None`, all connections are
            plotted.

        times : tuple of int or float | None (default None)
            Start and end times of the results to plot. If :obj:`None`, plot
            all times.

        n_rows : int (default ``1``)
            Number of rows of subplots per figure.

        n_cols : int (default ``1``)
            Number of columns of subplots per figure.

        major_tick_intervals : int | float (default ``500.0``)
            Intervals (in ms) at which the major ticks of the x- and y-axes
            should occur.

        minor_tick_intervals : int | float (default ``100.0``)
            Intervals (in ms) at which the minor ticks of the x- and y-axes
            should occur.

        show : bool (default ``True``)
            Whether or not to show the plotted results.

        Returns
        -------
        figures : list of matplotlib Figure
            Figures of the results in a list of length
            ``ceil(n_nodes / (n_rows * n_cols))``.

        axes : list of ~numpy.ndarray of matplotlib pyplot Axes
            Subplot axes for the results in a list of length
            ``ceil(n_nodes / (n_rows * n_cols))`` where each entry is a 1D
            :obj:`~numpy.ndarray` of length ``(n_rows * n_cols)``.

        Notes
        -----
        ``n_rows`` and ``n_cols`` of ``1`` will plot the results for each
        connection on a new figure.
        """
        figures, axes = self._plotting.plot(
            nodes=nodes,
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
            self._tau.append(self.times[self._data[node_i].argmax()])
        self._tau = tuple(self._tau)

    @property
    def tau(self) -> tuple[float]:
        """Return the estimated time delay for each connection (in ms)."""
        return self._tau


class ResultsWaveShape(_ResultsBase):
    """Class for storing wave shape results.

    Parameters
    ----------
    data : ~numpy.ndarray, shape of [nodes, f1s, f2s]
        Results to store.

    indices : tuple of int
        Indices of the channels in the results.

    f1s : ~numpy.ndarray, shape of [low frequencies]
        Low frequencies (in Hz) in the results.

    f2s : ~numpy.ndarray, shape of [low frequencies]
        High frequencies (in Hz) in the results.

    name : str
        Name of the results being stored.

    Methods
    -------
    plot :
        Plot the results.

    get_results :
        Return a copy of the results.

    Attributes
    ----------
    name : str
        Name of the results.

    indices : tuple of int
        Indices of the channels in the results.

    shape : tuple of int
        Shape of the results i.e. [nodes, f1s, f2s].

    n_nodes : int
        Number of channels in the results.

    f1s : ~numpy.ndarray, shape of [low frequencies]
        Low frequencies (in Hz) in the results.

    f2s : ~numpy.ndarray, shape of [low frequencies]
        High frequencies (in Hz) in the results.
    """

    def __repr__(self) -> str:
        """Return printable representation of the object."""
        return repr(
            f"<Result: {self.name} | [{self.n_nodes} nodes, "
            f"{len(self.f1s)} f1s, {len(self.f2s)} f2s]>"
        )

    def __init__(
        self,
        data: np.ndarray,
        indices: tuple[int],
        f1s: np.ndarray,
        f2s: np.ndarray,
        name: str,
    ) -> None:  # noqa D107
        super().__init__(data, 3, name)
        self._sort_init_inputs(indices, f1s, f2s)

        self._plotting = _PlotWaveShape(
            data=self._data,
            indices=self.indices,
            f1s=self.f1s,
            f2s=self.f2s,
            name=self.name,
        )

    def _sort_init_inputs(
        self, indices: tuple[int], f1s: np.ndarray, f2s: np.ndarray
    ) -> None:
        """Sort inputs to the object."""
        super()._sort_indices_channels(indices)
        super()._sort_freq_inputs(f1s, f2s)

    def get_results(self) -> np.ndarray:
        """Return a copy of the results.

        Returns
        -------
        results : ~numpy.ndarray, shape of [nodes, f1s, f2s]
            The results.
        """
        return self._data.copy()

    def plot(
        self,
        nodes: tuple[int] | None = None,
        f1s: tuple[int | float] | None = None,
        f2s: tuple[int | float] | None = None,
        n_rows: int = 1,
        n_cols: int = 1,
        major_tick_intervals: int | float = 5.0,
        minor_tick_intervals: int | float = 1.0,
        plot_absolute: bool = True,
        cbar_range_abs: tuple[float] | tuple[tuple[float]] | None = None,
        cbar_range_real: tuple[float] | tuple[tuple[float]] | None = None,
        cbar_range_imag: tuple[float] | tuple[tuple[float]] | None = None,
        cbar_range_phase: tuple[float] | tuple[tuple[float]] | None = None,
        show: bool = True,
    ) -> tuple[list[Figure], list[np.ndarray]]:
        """Plot the results.

        Parameters
        ----------
        nodes : tuple of int | None (default None)
            Indices of results of channels to plot. If :obj:`None`, plot
            results of all channels.

        f1s : tuple of int or float | None (default None)
            Start and end low frequencies of the results to plot, respectively.
            If :obj:`None`, plot all low frequencies.

        f2s : tuple of int or float | None (default None)
            Start and end high frequencies of the results to plot,
            respectively. If :obj:`None`, plot all high frequencies.

        n_rows : int (default ``1``)
            Number of rows of subplots per figure.

        n_cols : int (default ``1``)
            Number of columns of subplots per figure.

        major_tick_intervals : int | float (default ``5.0``)
            Intervals (in Hz) at which the major ticks of the x- and y-axes
            should occur.

        minor_tick_intervals : int | float (default ``1.0``)
            Intervals (in Hz) at which the minor ticks of the x- and y-axes
            should occur.

        plot_absolute : bool (default ``True``)
            Whether or not to plot the absolute values of the real and
            imaginary parts of the results.

        cbar_range_abs : tuple of float | tuple of tuple of float | None (default None)
            Range (in units of the data) for the colourbars of the absolute
            value of the results, consisting of the lower and upper limits,
            respectively. If :obj:`None`, the range is computed automatically.
            If a tuple of float, this range is used for all plots. If a tuple of
            tuple of float, the ranges are used for each individual plot.

        cbar_range_real : tuple of float | tuple of tuple of float | None (default None)
            Range (in units of the data) for the colourbars of the real value
            of the results, consisting of the lower and upper limits,
            respectively. If :obj:`None`, the range is computed automatically.
            If a tuple of float, this range is used for all plots. If a tuple of
            tuple of float, the ranges are used for each individual plot.

        cbar_range_imag : tuple of float | tuple of tuple of float | None (default None)
            Range (in units of the data) for the colourbars of the imaginary
            value of the results, consisting of the lower and upper limits,
            respectively. If :obj:`None`, the range is computed automatically.
            If a tuple of float, this range is used for all plots. If a tuple of
            tuple of float, the ranges are used for each individual plot.

        cbar_range_phase : tuple of float | tuple of tuple of float | None (default None)
            Range (in units of the data) for the colourbars of the phase of the
            results, consisting of the lower and upper limits, respectively. If
            :obj:`None`, the range is computed automatically. If a tuple of
            float, this range is used for all plots. If a tuple of tuple of
            float, the ranges are used for each individual plot. Note that
            results are limited to the range (-pi, pi].

        show : bool (default ``True``)
            Whether or not to show the plotted results.

        Returns
        -------
        figures : list of matplotlib Figure
            Figures of the results in a list of length
            ``ceil(n_nodes / (n_rows * n_cols))``.

        axes : list of ~numpy.ndarray of ~numpy.ndarray of matplotlib pyplot Axes
            Subplot axes for the results in a list of length
            ``ceil(n_nodes / (n_rows * n_cols))`` where each entry is a 1D
            :obj:`~numpy.ndarray` of length ``(n_rows * n_cols)``, whose
            entries are themselves 1D :obj:`~numpy.ndarray` of length 4,
            corresponding to the absolute, real, imaginary, and phase plots,
            respectively.

        Notes
        -----
        ``n_rows`` and ``n_cols`` of ``1`` will plot the results for each
        channel on a new figure.
        """  # noqa: E501
        figures, axes = self._plotting.plot(
            nodes=nodes,
            f1s=f1s,
            f2s=f2s,
            n_rows=n_rows,
            n_cols=n_cols,
            major_tick_intervals=major_tick_intervals,
            minor_tick_intervals=minor_tick_intervals,
            plot_absolute=plot_absolute,
            cbar_range_abs=cbar_range_abs,
            cbar_range_real=cbar_range_real,
            cbar_range_imag=cbar_range_imag,
            cbar_range_phase=cbar_range_phase,
            show=show,
        )

        return figures, axes
