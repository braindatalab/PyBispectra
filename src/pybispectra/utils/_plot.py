"""Private helper tools for plotting results."""

from abc import ABC, abstractmethod

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from matplotlib.ticker import ScalarFormatter, StrMethodFormatter

from pybispectra.utils._utils import _int_like, _number_like


class _PlotBase(ABC):
    """Base class for plotting results.

    Notes
    -----
    Does not check initialisation inputs, assuming these have been checked by the
    publicly-avaiable class/function.
    """

    f1s: np.ndarray = None
    f2s: np.ndarray = None
    times: np.ndarray | None = None

    def __init__(self, data: np.ndarray, indices: tuple, name: str) -> None:
        self._data = data
        self._indices = indices

        if len(indices) > 1 and np.all([isinstance(group, tuple) for group in indices]):
            self.n_nodes = len(indices[0])
        else:
            self.n_nodes = len(indices)

        self.name = name

    @abstractmethod
    def plot(self) -> None:
        """Plot the results."""

    @abstractmethod
    def _sort_plot_inputs(
        self,
        nodes: int | tuple[int] | None,
        n_rows: int,
        n_cols: int,
        major_tick_intervals: int | float,
        minor_tick_intervals: int | float,
    ) -> tuple[int]:
        """Sort the plotting inputs.

        Returns
        -------
        nodes : tuple of int
        """
        if nodes is None:
            nodes = tuple(range(self.n_nodes))
        if not isinstance(nodes, _int_like + (tuple,)):
            raise TypeError("`nodes` must be an int or tuple.")
        if isinstance(nodes, int):
            nodes = (nodes,)
        if not all(isinstance(con, _int_like) for con in nodes):
            raise TypeError("Entries of `nodes` must be ints.")
        if any(con >= self.n_nodes for con in nodes) or any(con < 0 for con in nodes):
            raise ValueError("The requested node is not present in the results.")

        if not isinstance(n_rows, _int_like) or not isinstance(n_cols, _int_like):
            raise TypeError("`n_rows` and `n_cols` must be integers.")
        if n_rows < 1 or n_cols < 1:
            raise ValueError("`n_rows` and `n_cols` must be >= 1.")

        if not isinstance(major_tick_intervals, _number_like) or not isinstance(
            minor_tick_intervals, _number_like
        ):
            raise TypeError(
                "`major_tick_intervals` and `minor_tick_intervals` should be ints or "
                "floats."
            )
        if major_tick_intervals <= 0 or minor_tick_intervals <= 0:
            raise ValueError(
                "`major_tick_intervals` and `minor_tick_intervals` should be > 0."
            )
        if minor_tick_intervals >= major_tick_intervals:
            raise ValueError(
                "`major_tick_intervals` should be > `minor_tick_intervals`."
            )

        return nodes

    def _sort_freq_inputs(
        self, f1s: np.ndarray, f2s: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Sort ``f1s`` and ``f2s`` inputs.

        Returns
        -------
        f1s : numpy.ndarray of float
            Low frequencies in the results to plot.

        f2s : numpy.ndarray of float
            High frequencies in the results to plot.

        f1_idcs : numpy.ndarray of int
            Indices of ``f1s`` in the results.

        f2_idcs : numpy.ndarray of int
            Indices of ``f2s`` in the results.
        """
        check_f1s = True
        check_f2s = True
        if f1s is None:
            f1s = self.f1s.copy()
            f1_idcs = np.arange(len(self.f1s), dtype=np.int32)
            check_f1s = False
        if f2s is None:
            f2s = self.f2s.copy()
            f2_idcs = np.arange(len(self.f1s), dtype=np.int32)
            check_f2s = False

        for freqs, check_freqs in zip([f1s, f2s], [check_f1s, check_f2s]):
            if check_freqs:
                if not isinstance(freqs, tuple):
                    raise TypeError("`f1s` and `f2s` must be tuples.")
                if len(freqs) != 2:
                    raise ValueError("`f1s` and `f2s` must have lengths of 2.")

        if check_f1s:
            f1_idcs = np.argwhere((self.f1s >= f1s[0]) & (self.f1s <= f1s[1])).T[0]
            if f1_idcs.size == 0:
                raise ValueError(
                    "No frequencies are present in the data for the range in `f1s`."
                )
            f1s = self.f1s[f1_idcs].copy()
        if check_f2s:
            f2_idcs = np.argwhere((self.f2s >= f2s[0]) & (self.f2s <= f2s[1])).T[0]
            if f2_idcs.size == 0:
                raise ValueError(
                    "No frequencies are present in the data for the range in `f2s`."
                )
            f2s = self.f2s[f2_idcs].copy()

        return f1s, f2s, f1_idcs, f2_idcs

    def _sort_time_inputs(
        self, times: tuple[int | float] | None
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        """Sort `times` input.

        Returns
        -------
        times : numpy.ndarray | None
            Times of the results to plot.

        time_idcs : numpy.ndarray | None
            Indices of times in ``times``.
        """
        if times is None or self.times is None:
            times = self.times
            time_idcs = (
                np.arange(times.size, dtype=np.int32) if times is not None else None
            )
        else:
            if not isinstance(times, tuple):
                raise TypeError("`times` must be a tuple.")
            if len(times) != 2:
                raise ValueError("`times` must have length of 2.")

            time_idcs = np.argwhere(
                (self.times >= times[0]) & (self.times <= times[1])
            ).T[0]
            if time_idcs.size == 0:
                raise ValueError(
                    "No timepoints are present in the data for the range in `times`."
                )
            times = self.times[time_idcs].copy()

        return times, time_idcs

    def _create_plots(
        self,
        nodes: tuple[int],
        n_rows: int,
        n_cols: int,
    ) -> tuple[list[Figure], list[np.ndarray]]:
        """Create figures and subplots to fill with results.

        Returns
        -------
        figures : list of matplotlib Figure
            Figures for the results in a list of length ``ceil(n_nodes / (n_rows *
            n_cols))``.

        axes : list of numpy.ndarray of matplotlib pyplot Axes
            Subplot axes for the results in a list of length ``ceil(n_nodes / (n_rows *
            n_cols))`` where each entry is a 1D ``numpy.ndarray`` of length ``(n_rows *
            n_cols)``.
        """
        figures = []
        axes = []

        plot_n = 0
        for node_i in range(len(nodes)):
            if node_i == plot_n:
                fig, axs = plt.subplots(n_rows, n_cols, layout="constrained")
                figures.append(fig)
                if n_rows * n_cols > 1:
                    axs = np.ravel(axs)
                else:
                    axs = np.array([axs])
                axes.append(axs)
                plot_n += n_rows * n_cols
            if plot_n >= len(nodes):
                break

        return figures, axes

    @abstractmethod
    def _plot_results(self) -> None:
        """Plot results on the relevant figures/subplots."""

    @abstractmethod
    def _set_axis_ticks(self) -> None:
        """Set major and minor tick intervals of x- and y-axes."""


class _PlotGeneral(_PlotBase):
    """Class for plotting general results."""

    def __init__(
        self,
        data: np.ndarray,
        indices: tuple[tuple[int]] | tuple[int],
        f1s: np.ndarray,
        f2s: np.ndarray,
        times: np.ndarray | None,
        name: str,
    ) -> None:  # noqa: D107
        super().__init__(data, indices, name)

        self.f1s = f1s.copy()
        self.f2s = f2s.copy()
        self.times = times.copy() if times is not None else None

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
        r"""Plot the results.

        Parameters
        ----------
        nodes : int | tuple[int] | None = None,
            Indices of channels to plot. If :obj:`None`, plot all channels.

        f1s : tuple of int or float | None (default None)
            Start and end low frequencies of the results to plot, respectively.
            If :obj:`None`, plot all low frequencies.

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
            Range (in units of pi) for the colourbars of the phase of the results,
            consisting of the lower and upper limits, respectively. If :obj:`None`, the
            range is computed automatically. If a tuple of float, this range is used for
            all plots. If a tuple of tuple of float, the ranges are used for each
            individual plot. Note that results should be limited to the range
            :math:`(0, 2$\\pi$]`.

        show : bool (default True)
            Whether or not to show the plotted results.

        Returns
        -------
        figures : list of matplotlib Figure
            Figures of the results in a list of length ``ceil(n_nodes / (n_rows *
            n_cols))``.

        axes : list of ~numpy.ndarray of ~numpy.ndarray of matplotlib pyplot Axes
            Subplot axes for the results in a list of length ``ceil(n_nodes / (n_rows *
            n_cols))`` where each entry is a 1D ``~numpy.ndarray`` of length ``(n_rows *
            n_cols)``, whose entries are themselves 1D ``~numpy.ndarray`` of length 4,
            corresponding to the absolute, real, imaginary, and phase plots,
            respectively.

        Notes
        -----
        ``n_rows`` and ``n_cols`` of ``1`` will plot the results for each node on a new
        figure.
        """
        (nodes, f1s, f2s, f1_idcs, f2_idcs, times, time_idcs, cbar_ranges) = (
            self._sort_plot_inputs(
                nodes,
                f1s,
                f2s,
                times,
                n_rows,
                n_cols,
                major_tick_intervals,
                minor_tick_intervals,
                plot_absolute,
                mirror_cbar_range,
                cbar_range_abs,
                cbar_range_real,
                cbar_range_imag,
                cbar_range_phase,
            )
        )
        figures, subfigures, axes = self._create_plots(nodes, n_rows, n_cols)
        figures, axes = self._plot_results(
            figures,
            subfigures,
            axes,
            nodes,
            f1s,
            f2s,
            f1_idcs,
            f2_idcs,
            times,
            time_idcs,
            n_rows,
            n_cols,
            major_tick_intervals,
            minor_tick_intervals,
            plot_absolute,
            mirror_cbar_range,
            cbar_ranges,
        )

        if show:  # pragma: no cover
            plt.show()

        return figures, axes

    def _sort_plot_inputs(
        self,
        nodes: int | tuple[int] | None,
        f1s: tuple[int | float] | None,
        f2s: tuple[int | float] | None,
        times: tuple[int | float] | None,
        n_rows: int,
        n_cols: int,
        major_tick_intervals: int | float,
        minor_tick_intervals: int | float,
        plot_absolute: bool,
        mirror_cbar_range: bool,
        cbar_range_abs: tuple[float] | list[tuple[float]] | None,
        cbar_range_real: tuple[float] | list[tuple[float]] | None,
        cbar_range_imag: tuple[float] | list[tuple[float]] | None,
        cbar_range_phase: tuple[float] | list[tuple[float]] | None,
    ) -> tuple[
        tuple[int],
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray | None,
        np.ndarray | None,
        list[list[tuple[float | None]]],
    ]:
        """Sort the plotting inputs.

        Returns
        -------
        nodes : tuple of int

        f1s : numpy.ndarray of float
            Low frequencies in the results to plot.

        f2s : numpy.ndarray of float
            High frequencies in the results to plot.

        f1_idcs : numpy.ndarray of int
            Indices of ``f1s`` in the results.

        f2_idcs : numpy.ndarray of int
            Indices of ``f2s`` in the results.

        times : numpy.ndarray of float
            Times in the results to plot.

        time_idcs : numpy.ndarray of int
            Indices of ``times`` in the results.

        cbar_ranges : list of list of tuple of float or None
            Colourbar ranges for the absolute, real, imaginary, and phase plots,
            respectively.
        """
        nodes = super()._sort_plot_inputs(
            nodes, n_rows, n_cols, major_tick_intervals, minor_tick_intervals
        )
        f1s, f2s, f1_idcs, f2_idcs = super()._sort_freq_inputs(f1s, f2s)
        times, time_idcs = super()._sort_time_inputs(times)

        if not isinstance(plot_absolute, bool):
            raise TypeError("`plot_absolute` must be a bool.")
        if not isinstance(mirror_cbar_range, bool):
            raise TypeError("`mirror_cbar_range` must be a bool.")

        cbar_ranges = [
            cbar_range_abs,
            cbar_range_real,
            cbar_range_imag,
            cbar_range_phase,
        ]
        cbar_names = ["abs", "real", "imag", "phase"]
        cbar_idx = 0
        for cbar_range, cbar_name in zip(cbar_ranges, cbar_names):
            if not isinstance(cbar_range, (list, tuple, type(None))):
                raise TypeError(
                    f"`cbar_range_{cbar_name}` must be a list, tuple, or None."
                )
            if isinstance(cbar_range, list):
                if len(cbar_range) != len(nodes):
                    raise ValueError(
                        f"If `cbar_range_{cbar_name}` is a list, one entry must be "
                        "provided for each node being plotted."
                    )
            else:
                fill = cbar_range if cbar_range is not None else [None, None]
                cbar_range = [fill for _ in range(len(nodes))]
            for entry in cbar_range:
                if len(entry) != 2:
                    raise ValueError(
                        f"Limits in `cbar_range_{cbar_name}` must have length of 2."
                    )
            cbar_ranges[cbar_idx] = cbar_range
            cbar_idx += 1

        return (nodes, f1s, f2s, f1_idcs, f2_idcs, times, time_idcs, cbar_ranges)

    def _create_plots(
        self, nodes: tuple[int], n_rows: int, n_cols: int
    ) -> tuple[list[Figure], list[np.ndarray]]:
        """Create figures and subplots to fill with results.

        Returns
        -------
        figures : list of matplotlib Figure
            Figures for the results in a list of length ``ceil(n_nodes / (n_rows *
            n_cols))``.

        subfigures : list of matplotlib Figure
            Subfigures for the results in a list of length ``figures``.

        axes : list of numpy.ndarray of numpy.ndarray of matplotlib pyplot Axes
            Subplot axes for the results in a list of length ``ceil(n_nodes / (n_rows *
            n_cols))`` where each entry is a 1D ``numpy.ndarray`` of length ``(n_rows *
            n_cols)``, whose entries are themselves 1D ``numpy.ndarray`` of length 4,
            corresponding to the absolute, real, imaginary, and phase plots,
            respectively.
        """
        figures = []
        subfigures = []
        axes = []

        plot_n = 0
        for node_i in range(len(nodes)):
            if node_i == plot_n:
                fig = plt.figure(layout="compressed")
                figures.append(fig)
                subfigs = fig.subfigures(n_rows, n_cols, wspace=0.05, hspace=0.05)
                if n_rows * n_cols > 1:
                    subfigs = np.ravel(subfigs)
                else:
                    subfigs = np.array([subfigs])
                subfigures.append(subfigs)
                subfig_axs = []
                for subfig in subfigs:
                    axs = np.ravel(subfig.subplots(2, 2))
                    subfig_axs.append(axs)
                axes.append(np.array(subfig_axs))
                plot_n += n_rows * n_cols
            if plot_n >= len(nodes):
                break

        return figures, subfigures, axes

    def _plot_results(
        self,
        figures: list[Figure],
        subfigures: list[np.ndarray],
        axes: list[np.ndarray],
        nodes: tuple[int],
        f1s: np.ndarray,
        f2s: np.ndarray,
        f1_idcs: np.ndarray,
        f2_idcs: np.ndarray,
        times: np.ndarray | None,
        time_idcs: np.ndarray | None,
        n_rows: int,
        n_cols: int,
        major_tick_intervals: int | float,
        minor_tick_intervals: int | float,
        plot_absolute: bool,
        mirror_cbar_range: bool,
        cbar_ranges: list[list[tuple[float | None]]],
    ) -> tuple[list[Figure], list[np.ndarray]]:
        """Plot results on the relevant figures/subplots."""
        fig_i = 0
        plot_n = 0
        fig_plot_n = 0
        stop_plotting = False
        while not stop_plotting:
            for _ in range(n_rows):
                for _ in range(n_cols):
                    node_i = nodes[plot_n]
                    subfig = subfigures[fig_i][fig_plot_n]

                    data_funcs = [np.abs, np.real, np.imag, np.angle]
                    axes_titles = ["Absolute", "Real", "Imaginary", "Phase"]
                    cmaps = ["viridis", "viridis", "viridis", "twilight_shifted"]
                    cbar_titles = [
                        "Magnitude (A.U.)",
                        "Magnitude (A.U.)",
                        "Magnitude (A.U.)",
                        "Phase (radians)",
                    ]
                    for (
                        axis,
                        data_func,
                        axis_title,
                        cmap,
                        cbar_title,
                        cbar_range,
                    ) in zip(
                        axes[fig_i][fig_plot_n],
                        data_funcs,
                        axes_titles,
                        cmaps,
                        cbar_titles,
                        cbar_ranges,
                    ):
                        if axis_title in ["Imaginary", "Phase"] and np.all(
                            np.isreal(self._data[node_i])
                        ):
                            # If data is real, np.imag and np.angle return 0, resulting
                            # in coloured panels, so instead set to NaN for empty panels
                            data = np.full_like(
                                self._data[node_i][np.ix_(f1_idcs, f2_idcs)],
                                fill_value=np.nan,
                            )
                        else:
                            data = data_func(
                                self._data[node_i][np.ix_(f1_idcs, f2_idcs)]
                            )
                        data = np.moveaxis(data, 0, 1)

                        if time_idcs is not None:
                            assert data.ndim == 3, (
                                "PyBispectra Internal Error: data to plot for a given "
                                "node should be 3D prior to aggregating over time. "
                                "Please contact the PyBispectra developers."
                            )
                            data = data[..., time_idcs].mean(axis=-1)
                        else:
                            assert data.ndim == 2, (
                                "PyBispectra Internal Error: data to plot for a given "
                                "node should be 2D prior to aggregating over time. "
                                "Please contact the PyBispectra developers."
                            )

                        if axis_title in ["Real", "Imaginary"]:
                            if plot_absolute:
                                data = np.abs(data)
                                axis_title = f"|{axis_title}|"
                            elif mirror_cbar_range and cbar_range[plot_n] == [
                                None,
                                None,
                            ]:
                                max_abs_data = np.nanmax(np.abs(data))
                                cbar_range[plot_n] = [-max_abs_data, max_abs_data]

                        if axis_title == "Phase":
                            # np.angle returns values in range (-pi, pi]; nice to
                            # convert range to (0, 2*pi]
                            data[data < 0] += 2 * np.pi
                            data /= np.pi  # normalise units of the data
                            format_ = StrMethodFormatter(r"{x} $\pi$")
                        else:
                            format_ = ScalarFormatter()

                        mesh = axis.pcolormesh(
                            f1s,
                            f2s,
                            data,
                            cmap=cmap,
                            vmin=cbar_range[plot_n][0],
                            vmax=cbar_range[plot_n][1],
                        )

                        plt.colorbar(
                            mesh,
                            ax=axis,
                            label=cbar_title,
                            shrink=0.5,
                            format=format_,
                            panchor=False,
                        )

                        axis.set_title(axis_title)
                        axis.set_aspect("equal")
                        self._set_axis_ticks(
                            axis, major_tick_intervals, minor_tick_intervals
                        )
                        axis.grid(
                            which="major",
                            axis="both",
                            linestyle="--",
                            color=[0.7, 0.7, 0.7],
                            alpha=0.7,
                        )
                        axis.set_xlabel("$f_1$ (Hz)")
                        axis.set_ylabel("$f_2$ (Hz)")

                    subfig.suptitle(self._get_axis_title(node_i, times))

                    plot_n += 1
                    fig_plot_n += 1
                    if fig_plot_n >= n_rows * n_cols or fig_plot_n >= len(nodes):
                        figures[fig_i].suptitle(self.name)
                    if fig_plot_n >= n_rows * n_cols:
                        # move to next figure
                        fig_plot_n = 0
                        fig_i += 1
                    if fig_i >= len(figures) or fig_plot_n >= len(nodes):
                        # stop plotting
                        if fig_i < len(figures):
                            # remove excess axes from current figure
                            for axis_array in axes[fig_i][fig_plot_n:]:
                                [axis.remove() for axis in axis_array]
                            axes[fig_i] = np.delete(
                                axes[fig_i],
                                np.arange(fig_plot_n, n_rows * n_cols),
                                axis=0,
                            )

                        stop_plotting = True
                    if stop_plotting:
                        break
                if stop_plotting:
                    break

        return figures, axes

    def _set_axis_ticks(
        self,
        axis: plt.Axes,
        major_tick_intervals: int | float,
        minor_tick_intervals: int | float,
    ) -> None:
        """Set major and minor tick intervals of x- and y-axes."""
        axis.xaxis.set_major_locator(plt.MultipleLocator(major_tick_intervals))
        axis.xaxis.set_minor_locator(plt.MultipleLocator(minor_tick_intervals))
        axis.yaxis.set_major_locator(plt.MultipleLocator(major_tick_intervals))
        axis.yaxis.set_minor_locator(plt.MultipleLocator(minor_tick_intervals))

    def _get_axis_title(self, node_i: int, times: np.ndarray | None) -> str:
        """Get title for the axis.

        Parameters
        ----------
        node_i : int
            Index of the node being plotted.

        times : numpy.ndarray | None
            Timepoints of the results being plotted.

        Returns
        -------
        title : str
            Title of the axis.
        """
        title = (
            f"k: {self._indices[0][node_i]} | m: {self._indices[1][node_i]} | "
            f"n: {self._indices[2][node_i]}"
        )
        if times is not None:
            title += f" | {times[0]:.3f} - {times[-1]:.3f} s"

        return title


class _PlotCFC(_PlotBase):
    """Class for plotting cross-frequency coupling (CFC) results."""

    def __init__(
        self,
        data: np.ndarray,
        indices: tuple[tuple[int]],
        f1s: np.ndarray,
        f2s: np.ndarray,
        times: np.ndarray | None,
        name: str,
    ) -> None:  # noqa: D107
        super().__init__(data, indices, name)

        self.f1s = f1s.copy()
        self.f2s = f2s.copy()
        self.times = times.copy() if times is not None else None

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
            Indices of nodes to plot. If :obj:`None`, plot all nodes.

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
        ``n_rows`` and ``n_cols`` of ``1`` will plot the results for each node on a new
        figure.
        """  # noqa: E501
        nodes, f1s, f2s, f1_idcs, f2_idcs, times, time_idcs, cbar_range = (
            self._sort_plot_inputs(
                nodes,
                f1s,
                f2s,
                times,
                n_rows,
                n_cols,
                major_tick_intervals,
                minor_tick_intervals,
                cbar_range,
            )
        )
        figures, axes = self._create_plots(nodes, n_rows, n_cols)
        figures, axes = self._plot_results(
            figures,
            axes,
            nodes,
            f1s,
            f2s,
            f1_idcs,
            f2_idcs,
            times,
            time_idcs,
            n_rows,
            n_cols,
            major_tick_intervals,
            minor_tick_intervals,
            cbar_range,
        )

        if show:  # pragma: no cover
            plt.show()

        return figures, axes

    def _sort_plot_inputs(
        self,
        nodes: int | tuple[int] | None,
        f1s: tuple[int | float] | None,
        f2s: tuple[int | float] | None,
        times: tuple[int | float] | None,
        n_rows: int,
        n_cols: int,
        major_tick_intervals: int | float,
        minor_tick_intervals: int | float,
        cbar_range: tuple[float] | list[tuple[float]] | None,
    ) -> tuple[
        tuple[int],
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray | None,
        np.ndarray | None,
        tuple[tuple[float | None]],
    ]:
        """Sort the plotting inputs.

        Returns
        -------
        nodes : tuple of int

        f1s : numpy.ndarray of float
            Low frequencies in the results to plot.

        f2s : numpy.ndarray of float
            High frequencies in the results to plot.

        f1_idcs : numpy.ndarray of int
            Indices of ``f1s`` in the results.

        f2_idcs : numpy.ndarray of int
            Indices of ``f2s`` in the results.

        times : numpy.ndarray of float | None
            Times in the results to plot.

        time_idcs : numpy.ndarray of int | None
            Indices of ``times`` in the results.

        cbar_range : list of tuple of float or None
        """
        nodes = super()._sort_plot_inputs(
            nodes, n_rows, n_cols, major_tick_intervals, minor_tick_intervals
        )
        f1s, f2s, f1_idcs, f2_idcs = super()._sort_freq_inputs(f1s, f2s)
        times, time_idcs = super()._sort_time_inputs(times)

        if not isinstance(cbar_range, (list, tuple, type(None))):
            raise TypeError("`cbar_range` must be a list, tuple, or None.")
        if isinstance(cbar_range, list):
            if len(cbar_range) != len(nodes):
                raise ValueError(
                    "If `cbar_range` is a list, one entry must be provided for each "
                    "node being plotted."
                )
        else:
            fill = cbar_range if cbar_range is not None else [None, None]
            cbar_range = [fill for _ in range(len(nodes))]
        for entry in cbar_range:
            if len(entry) != 2:
                raise ValueError("Limits in `cbar_range` must have length of 2.")

        return nodes, f1s, f2s, f1_idcs, f2_idcs, times, time_idcs, cbar_range

    def _plot_results(
        self,
        figures: list[Figure],
        axes: list[np.ndarray],
        nodes: tuple[int],
        f1s: np.ndarray,
        f2s: np.ndarray,
        f1_idcs: np.ndarray,
        f2_idcs: np.ndarray,
        times: np.ndarray | None,
        time_idcs: np.ndarray | None,
        n_rows: int,
        n_cols: int,
        major_tick_intervals: int | float,
        minor_tick_intervals: int | float,
        cbar_range: list[tuple[float | None]],
    ) -> tuple[list[Figure], list[np.ndarray]]:
        """Plot results on the relevant figures/subplots."""
        fig_i = 0
        plot_n = 0
        fig_plot_n = 0
        stop_plotting = False
        while not stop_plotting:
            for _ in range(n_rows):
                for _ in range(n_cols):
                    node_i = nodes[plot_n]
                    axis = axes[fig_i][fig_plot_n]

                    data = self._data[node_i][np.ix_(f1_idcs, f2_idcs)]
                    data = np.moveaxis(data, 0, 1)
                    if time_idcs is not None:
                        assert data.ndim == 3, (
                            "PyBispectra Internal Error: data to plot for a given node "
                            "should be 3D prior to aggregating over time. Please "
                            "contact the PyBispectra developers."
                        )
                        data = data[..., time_idcs].mean(axis=-1)
                    else:
                        assert data.ndim == 2, (
                            "PyBispectra Internal Error: data to plot for a given node "
                            "should be 2D prior to aggregating over time. Please "
                            "contact the PyBispectra developers."
                        )

                    mesh = axis.pcolormesh(
                        f1s,
                        f2s,
                        data,
                        vmin=cbar_range[plot_n][0],
                        vmax=cbar_range[plot_n][1],
                    )

                    plt.colorbar(mesh, ax=axis, label="Coupling (A.U.)", shrink=0.3)

                    axis.set_aspect("equal")
                    self._set_axis_ticks(
                        axis, major_tick_intervals, minor_tick_intervals
                    )
                    axis.grid(
                        which="major",
                        axis="both",
                        linestyle="--",
                        color=[0.7, 0.7, 0.7],
                        alpha=0.7,
                    )
                    axis.set_xlabel("$f_1$ (Hz)")
                    axis.set_ylabel("$f_2$ (Hz)")
                    axis.set_title(self._get_axis_title(node_i, times))

                    plot_n += 1
                    fig_plot_n += 1
                    if fig_plot_n >= n_rows * n_cols or fig_plot_n >= len(nodes):
                        figures[fig_i].suptitle(self.name)
                    if fig_plot_n >= n_rows * n_cols:
                        # move to next figure
                        fig_plot_n = 0
                        fig_i += 1
                    if fig_i >= len(figures) or fig_plot_n >= len(nodes):
                        # stop plotting
                        if fig_i < len(figures):
                            # remove excess axes from current figure
                            for axis in axes[fig_i][fig_plot_n:]:
                                axis.remove()
                            axes[fig_i] = np.delete(
                                axes[fig_i],
                                np.arange(fig_plot_n, n_rows * n_cols),
                                axis=0,
                            )

                        stop_plotting = True
                    if stop_plotting:
                        break
                if stop_plotting:
                    break

        return figures, axes

    def _set_axis_ticks(
        self,
        axis: plt.Axes,
        major_tick_intervals: int | float,
        minor_tick_intervals: int | float,
    ) -> None:
        """Set major and minor tick intervals of x- and y-axes."""
        axis.xaxis.set_major_locator(plt.MultipleLocator(major_tick_intervals))
        axis.xaxis.set_minor_locator(plt.MultipleLocator(minor_tick_intervals))
        axis.yaxis.set_major_locator(plt.MultipleLocator(major_tick_intervals))
        axis.yaxis.set_minor_locator(plt.MultipleLocator(minor_tick_intervals))

    def _get_axis_title(self, node_i: int, times: np.ndarray | None) -> str:
        """Get title for the axis.

        Parameters
        ----------
        node_i : int
            Index of the node being plotted.

        times : numpy.ndarray | None
            Timepoints of the results being plotted.

        Returns
        -------
        title : str
            Title of the axis.
        """
        title = f"Seed: {self._indices[0][node_i]} | Target: {self._indices[1][node_i]}"
        if times is not None:
            title += f" | {times[0]:.3f} - {times[-1]:.3f} s"

        return title


class _PlotTDE(_PlotBase):
    """Class for plotting time-delay estimation (TDE) results."""

    def __init__(
        self,
        data: np.ndarray,
        tau: tuple[float],
        indices: tuple[tuple[int]],
        freq_bands: tuple[tuple[float]] | None,
        times: np.ndarray,
        name: str,
    ) -> None:  # noqa: D107
        super().__init__(data, indices, name)

        self.tau = tau
        self.freq_bands = freq_bands
        self.times = times.copy()

        if self.freq_bands is None:
            self._n_fbands = self._data.shape[1]
        else:
            self._n_fbands = len(self.freq_bands)

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
            Indices of connections to plot. If :obj:`None`, plot all connections.

        freq_bands : int | tuple of int | None (default None)
            Indices of frequency bands to plot. If :obj:`None`, all frequency bands are
            plotted.

        times : tuple of int or float | None (default None)
            Start and end times of the results to plot, respectively. If :obj:`None`,
            plot all times.

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
            n_cols))`` where each entry is a 1D ``~numpy.ndarray`` of length ``(n_rows *
            n_cols)``.

        Notes
        -----
        ``n_rows`` and ``n_cols`` of ``1`` will plot the results for each connection on
        a new figure.
        """
        nodes, freq_bands, times, time_idcs = self._sort_plot_inputs(
            nodes,
            freq_bands,
            times,
            n_rows,
            n_cols,
            major_tick_intervals,
            minor_tick_intervals,
        )
        figures, axes = self._create_plots(nodes, freq_bands, n_rows, n_cols)
        figures, axes = self._plot_results(
            figures,
            axes,
            nodes,
            freq_bands,
            times,
            time_idcs,
            n_rows,
            n_cols,
            major_tick_intervals,
            minor_tick_intervals,
        )

        if show:  # pragma: no cover
            plt.show()

        return figures, axes

    def _sort_plot_inputs(
        self,
        nodes: int | tuple[int] | None,
        freq_bands: int | tuple[int] | None,
        times: tuple[int | float] | None,
        n_rows: int,
        n_cols: int,
        major_tick_intervals: float,
        minor_tick_intervals: float,
    ) -> tuple[tuple[int], tuple[int], np.ndarray, np.ndarray]:
        """Sort the plotting inputs.

        Returns
        -------
        nodes : tuple of int

        freq_bands : tuple of int

        times : numpy.ndarray
            Times of the results to plot.

        time_idcs : numpy.ndarray
            Indices of times in ``times``.
        """
        nodes = super()._sort_plot_inputs(
            nodes,
            n_rows,
            n_cols,
            major_tick_intervals,
            minor_tick_intervals,
        )
        freq_bands = self._sort_freq_band_inputs(freq_bands)
        times, time_idcs = super()._sort_time_inputs(times)

        return nodes, freq_bands, times, time_idcs

    def _sort_freq_band_inputs(self, freq_bands: int | tuple[int] | None) -> tuple[int]:
        """Sort `freq_bands` input."""
        if freq_bands is None:
            freq_bands = tuple(range(self._n_fbands))
        else:
            if not isinstance(freq_bands, _int_like + (tuple,)):
                raise TypeError("`freq_bands` must be an int or tuple.")
            if isinstance(freq_bands, _int_like):
                freq_bands = (freq_bands,)
            if not all(isinstance(fband, _int_like) for fband in freq_bands):
                raise TypeError("Entries of `freq_bands` must be ints.")
            if any(fband >= self._n_fbands for fband in freq_bands) or any(
                fband < 0 for fband in freq_bands
            ):
                raise ValueError(
                    "The requested frequency band is not present in the results."
                )

        return freq_bands

    def _create_plots(
        self,
        nodes: tuple[int],
        freq_bands: tuple[int],
        n_rows: int,
        n_cols: int,
    ) -> tuple[list[Figure], list[np.ndarray]]:
        """Create figures and subplots to fill with results.

        Returns
        -------
        figures : list of matplotlib Figure
            Figures for the results in a list of length ``ceil(n_nodes / (n_rows *
            n_cols))``.

        axes : list of numpy.ndarray of matplotlib pyplot Axes
            Subplot axes for the results in a list of length ``ceil(n_nodes / (n_rows *
            n_cols))`` where each entry is a 1D ``numpy.ndarray`` of length ``(n_rows *
            n_cols)``.
        """
        figures = []
        axes = []

        plot_n = 0
        for plot_i in range(len(nodes) * len(freq_bands)):
            if plot_i == plot_n:
                fig, axs = plt.subplots(n_rows, n_cols, layout="constrained")
                figures.append(fig)
                if n_rows * n_cols > 1:
                    axs = np.ravel(axs)
                else:
                    axs = np.array([axs])
                axes.append(axs)
                plot_n += n_rows * n_cols
            if plot_n >= len(nodes) * len(freq_bands):
                break

        return figures, axes

    def _plot_results(
        self,
        figures: list[Figure],
        axes: list[np.ndarray],
        nodes: tuple[int],
        freq_bands: tuple[int],
        times: np.ndarray,
        time_idcs: np.ndarray,
        n_rows: int,
        n_cols: int,
        major_tick_intervals: int | float,
        minor_tick_intervals: int | float,
    ) -> tuple[list[Figure], list[np.ndarray]]:
        """Plot results on the relevant figures/subplots."""
        fig_i = 0
        node_n = 0
        fband_n = 0
        fig_plot_n = 0
        stop_plotting = False
        while not stop_plotting:
            for _ in range(n_rows):
                for _ in range(n_cols):
                    node_i = nodes[node_n]
                    fband_i = freq_bands[fband_n]
                    axis = axes[fig_i][fig_plot_n]

                    axis.plot(times, self._data[node_i, fband_i, time_idcs])

                    self._mark_delay(
                        axis,
                        times,
                        self.tau[node_i, fband_i],
                        self._data[node_i, fband_i, time_idcs],
                    )

                    self._set_axis_ticks(
                        axis, major_tick_intervals, minor_tick_intervals
                    )
                    axis.grid(
                        which="major",
                        axis="x",
                        linestyle="-",
                        color=[0.7, 0.7, 0.7],
                        alpha=0.7,
                    )
                    axis.set_xlabel("Time (ms)")
                    axis.set_ylabel("Estimate strength (A.U.)")

                    axis.set_title(self._get_axis_title(node_i, fband_i))

                    fband_n += 1
                    fig_plot_n += 1
                    if fband_n >= len(freq_bands):
                        fband_n = 0
                        node_n += 1
                    if fig_plot_n >= n_rows * n_cols or fig_plot_n >= len(nodes) * len(
                        freq_bands
                    ):
                        figures[fig_i].suptitle(self.name)
                    if fig_plot_n >= n_rows * n_cols:
                        # move to next figure
                        fig_plot_n = 0
                        fig_i += 1
                    if fig_i >= len(figures) or fig_plot_n >= len(nodes) * len(
                        freq_bands
                    ):
                        # stop plotting
                        if fig_i < len(figures):
                            # remove excess axes from current figure
                            for axis in axes[fig_i][fig_plot_n:]:
                                axis.remove()
                            axes[fig_i] = np.delete(
                                axes[fig_i],
                                np.arange(fig_plot_n, n_rows * n_cols),
                                axis=0,
                            )

                        stop_plotting = True
                    if stop_plotting:
                        break
                if stop_plotting:
                    break

        return figures, axes

    def _mark_delay(
        self, axis: plt.Axes, times: np.ndarray, tau: float, results: np.ndarray
    ) -> None:
        """Mark estimated delay on the plot."""
        if tau not in times:
            xlim = axis.get_xlim()
            ylim = axis.get_ylim()
            xrange = xlim[1] - xlim[0]
            yrange = ylim[1] - ylim[0]
            annot_xy = ((xrange / 2) + xlim[0], ylim[1] - yrange * 0.05)
            alignment = "center"
        else:
            tau_idx = np.where(times == tau)[0][0]
            annot_xy = (tau, results[tau_idx])
            alignment = "left"
        axis.annotate(f"$\\tau$ = {tau:.2f} ms", xy=annot_xy, ha=alignment)

    def _set_axis_ticks(
        self,
        axis: plt.Axes,
        major_tick_intervals: int | float,
        minor_tick_intervals: int | float,
    ) -> None:
        """Set major and minor tick intervals of the x-axis."""
        axis.xaxis.set_major_locator(plt.MultipleLocator(major_tick_intervals))
        axis.xaxis.set_minor_locator(plt.MultipleLocator(minor_tick_intervals))

    def _get_axis_title(self, node_i: int, fband_i: int) -> str:
        """Get title for the axis.

        Parameters
        ----------
        node_i : int
            Index of the connection being plotted.

        fband_i : int
            Index of the frequency band being plotted.

        Returns
        -------
        title : str
            Title of the axis.
        """
        title = (
            f"Seed: {self._indices[0][node_i]} | Target: {self._indices[1][node_i]} | "
        )
        if self.freq_bands is not None:
            title += (
                f"{self.freq_bands[fband_i][0]:.2f} - "
                f"{self.freq_bands[fband_i][1]:.2f} Hz"
            )
        else:
            title += f"Band {fband_i + 1}"

        return title


class _PlotWaveShape(_PlotGeneral):
    """Class for plotting waveshape results."""

    def _get_axis_title(self, node_i: int, times: np.ndarray | None) -> str:
        """Get title for the axis.

        Parameters
        ----------
        node_i : int
            Index of the node being plotted.

        times : np.ndarray | None
            Timepoints of the results being plotted, respectively.

        Returns
        -------
        title : str
            Title of the axis.
        """
        title = f"Channel: {self._indices[node_i]}"
        if times is not None:
            title += f" | {times[0]:.3f} - {times[-1]:.3f} s"

        return title
