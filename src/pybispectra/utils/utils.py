"""Helper tools for processing and storing results."""

import copy
from warnings import warn

from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from numba import njit
import numpy as np
from pqdm.processes import pqdm
import scipy as sp


class ResultsCFC:
    """Class for storing cross-frequency coupling (CFC) results.

    Parameters
    ----------
    data : numpy.ndarray of float
        3D array of results to store with shape `[connections x f1 x f2]`.

    indices : tuple of numpy.ndarray of int
        Indices of the channels for each connection of :attr:`data`. Should
        contain 2 1D arrays of equal length for the seed and target indices,
        respectively.

    f1 : numpy.ndarray of float
        1D array of low frequencies in :attr:`data`.

    f2 : numpy.ndarray of float
        1D array of high frequencies in :attr:`data`.

    name : str
        Name of the results being stored.

    Attributes
    ----------
    name : str
        Name of the results.

    indices : tuple of numpy.ndarray of int
        Indices of the channels for each connection of the results. Contains
        two 1D arrays of equal length for the seed and target indices,
        respectively.

    n_cons : str
        Number of connections in the results.

    f1 : numpy.ndarray of float
        1D array of low frequencies in the results.

    f2 : numpy.ndarray of float
        1D array of high frequencies in the results.
    """

    _data = None

    indices = None
    _seeds = None
    _targets = None
    n_cons = None
    _n_chans = None

    f1 = None
    f2 = None

    name = None

    def __repr__(self) -> str:
        """Return printable represenation of the object."""
        return repr(
            f"<Result: {self.name} | [{self.n_cons} connections x "
            f"{len(self.f1)} f1 x {len(self.f2)} f2]>"
        )

    def __init__(
        self,
        data: np.ndarray,
        indices: tuple[np.ndarray],
        f1: np.ndarray,
        f2: np.ndarray,
        name: str,
    ) -> None:
        self._sort_init_inputs(data, indices, f1, f2, name)

    def _sort_init_inputs(
        self,
        data: np.ndarray,
        indices: tuple[np.ndarray],
        f1: np.ndarray,
        f2: np.ndarray,
        name: str,
    ) -> None:
        """Sort inputs to the object."""
        if not isinstance(data, np.ndarray):
            raise TypeError("`data` must be a NumPy array.")
        if data.ndim != 3:
            raise ValueError("`data` must be a 3D array.")
        self._data = data.copy()

        if not isinstance(indices, tuple):
            raise TypeError("`indices` must be a tuple.")
        if len(indices) != 2:
            raise ValueError("`indices` must have a length of 2.")
        if not isinstance(indices[0], np.ndarray) or not isinstance(
            indices[1], np.ndarray
        ):
            raise TypeError("Entries of `indices` must be NumPy arrays.")
        if indices[0].ndim != 1 or indices[1].ndim != 1:
            raise ValueError("Entries of `indices` must be 1D arrays.")
        if len(indices[0]) != len(indices[1]):
            raise ValueError("Entries of `indices` must have the same length.")
        self.indices = copy.copy(indices)
        self._seeds = indices[0].copy()
        self._targets = indices[1].copy()
        self.n_cons = len(indices[0])
        self._n_chans = len(np.unique([*self._seeds, *self._targets]))

        if not isinstance(f1, np.ndarray) or not isinstance(f2, np.ndarray):
            raise TypeError("`f1` and `f2` must be NumPy arrays.")
        if f1.ndim != 1 or f2.ndim != 1:
            raise ValueError("`f1` and `f2` must be 1D arrays.")
        self.f1 = f1.copy()
        self.f2 = f2.copy()

        if data.shape != (len(indices[0]), len(f1), len(f2)):
            raise ValueError("`data` must have shape [connections x f1 x f2].")

        if not isinstance(name, str):
            raise TypeError("`name` must be a string.")
        self.name = copy.copy(name)

    def get_results(
        self, form: str = "raveled"
    ) -> np.ndarray | tuple[np.ndarray, tuple[np.ndarray]]:
        """Return a copy of the results as arrays.

        Parameters
        ----------
        form : str (default ``"raveled"``)
            How the results should be returned: ``"raveled"`` - results have
            shape `[connections x f1 x f2]`; ``"compact"`` - results have shape
            `[seeds x targets x f1 x f2]`.

        Returns
        -------
        results : numpy.ndarray of float
            Spectral coupling results.

        indices : tuple of numpy.ndarray of int
            Channel indices of the seeds and targets. Only returned if ``form``
            is ``"compact"``.
        """
        accepted_forms = ["raveled", "compact"]
        if form not in accepted_forms:
            raise ValueError("`form` is not recognised.")

        if form == "raveled":
            return self._data.copy()
        return self._get_compact_results()

    def _get_compact_results(
        self,
    ) -> tuple[np.ndarray, tuple[np.ndarray]]:
        """Return a compacted form of the results.

        RETURNS
        -------
        compact_results : (default None) of float
            Results with shape `[seeds x targets x f1 x f2]`.

        indices : tuple[numpy.ndarray] of int
            Channel indices of ``compact_results`` for the seeds and targets,
            respectively.
        """
        fill = np.full((len(self.f1), len(self.f2)), fill_value=np.nan)
        compact_results = np.full(
            (self._n_chans, self._n_chans, len(self.f1), len(self.f2)),
            fill_value=fill,
        )
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
        for col_i, col in enumerate(compact_results.transpose(1, 0, 2, 3)):
            if not all(np.isnan(entry).all() for entry in col):
                filled_cols.append(col_i)
        compact_results = compact_results[np.ix_(filled_rows, filled_cols)]

        indices = (np.unique(self._seeds), np.unique(self._targets))

        return compact_results.copy(), indices

    def plot(
        self,
        connections: list[int] | None = None,
        f1: np.ndarray | None = None,
        f2: np.ndarray | None = None,
        n_rows: int = 1,
        n_cols: int = 1,
        major_tick_intervals: float = 5.0,
        minor_tick_intervals: float = 1.0,
        show: bool = True,
    ) -> tuple[list[Figure], list[np.ndarray]]:
        """Plot the results.

        Parameters
        ----------
        connections : list of int | None (default None)
            Indices of connections to plot. If ``None``, plot all connections.

        f1 : numpy.ndarray of float | None (default None)
            Low frequencies of the results to plot. If ``None``, plot all low
            frequencies.

        f2 : numpy.ndarray of float | None (default None)
            High frequencies of the results to plot. If ``None``, plot all high
            frequencies.

        n_rows : int (default ``1``)
            Number of rows of subplots per figure.

        n_cols : int (default ``1``)
            Number of columns of subplots per figure.

        major_tick_intervals : float (default ``5.0``)
            Intervals (in Hz) at which the major ticks of the x- and y-axes
            should occur.

        minor_tick_intervals : float (default ``1.0``)
            Intervals (in Hz) at which the minor ticks of the x- and y-axes
            should occur.

        show : bool (default ``True``)
            Whether or not to show the plotted results.

        Returns
        -------
        figures : list of matplotlib Figure
            Figures of the results in a list of length
            ``ceil(n_cons / (n_rows * n_cols))``.

        axes : list of numpy.ndarray of matplotlib pyplot Axes
            Subplot axes for the results in a list of length
            ``ceil(n_cons / (n_rows * n_cols))`` where each entry is a 1D
            ``numpy.ndarray`` of length ``(n_rows * n_cols)``.

        Notes
        -----
        ``n_rows`` and ``n_cols`` of ``1`` will plot the results for each
        connection on a new figure.
        """
        connections, f1, f2, f1_idcs, f2_idcs = self._sort_plot_inputs(
            connections,
            f1,
            f2,
            n_rows,
            n_cols,
            major_tick_intervals,
            minor_tick_intervals,
        )
        figures, axes = self._create_plots(connections, n_rows, n_cols)
        self._plot_results(
            figures,
            axes,
            connections,
            f1,
            f2,
            f1_idcs,
            f2_idcs,
            n_rows,
            n_cols,
            major_tick_intervals,
            minor_tick_intervals,
        )

        if show:
            plt.show()

        return figures, axes

    def _sort_plot_inputs(
        self,
        connections: list[int] | None,
        f1: np.ndarray | None,
        f2: np.ndarray | None,
        n_rows: int,
        n_cols: int,
        major_tick_intervals: float,
        minor_tick_intervals: float,
    ) -> tuple[list[int], np.ndarray, np.ndarray, list[int], list[int],]:
        """Sort the plotting inputs.

        Returns
        -------
        connections : list of int
            Indices of connections to plot.

        f1 : numpy.ndarray of float
            Low frequencies of the results to plot.

        f2 : numpy.ndarray of float
            High frequencies of the results to plot.

        f1_idcs : list of int
            Indices of ``f1`` in the results.

        f2_idcs : list of int
            Indices of ``f2`` in the results.
        """
        if connections is None:
            connections = np.arange(self.n_cons).tolist()
        if not isinstance(connections, list) or not all(
            isinstance(con, int) for con in connections
        ):
            raise TypeError("`connections` must be a list of integers.")
        if any(con >= self.n_cons for con in connections) or any(
            con < 0 for con in connections
        ):
            raise ValueError(
                "The requested connection is not present in the results."
            )

        if f1 is None:
            f1 = self.f1.copy()
        if f2 is None:
            f2 = self.f2.copy()
        if not isinstance(f1, np.ndarray) or not isinstance(f2, np.ndarray):
            raise TypeError("`f1` and `f2` must be NumPy arrays.")
        if f1.ndim != 1 or f2.ndim != 1:
            raise ValueError("`f1` and `f2` must be 1D arrays.")
        if any(freq not in self.f1 for freq in f1) or any(
            freq not in self.f2 for freq in f2
        ):
            raise ValueError(
                "Entries of `f1` and `f2` must be present in the results."
            )
        f1_idcs = [fast_find_first(self.f1, freq) for freq in f1]
        f2_idcs = [fast_find_first(self.f2, freq) for freq in f2]

        if not isinstance(n_rows, int) or not isinstance(n_cols, int):
            raise TypeError("`n_rows` and `n_cols` must be integers.")
        if n_rows < 1 or n_cols < 1:
            raise ValueError("`n_rows` and `n_cols` must be >= 1.")

        if not isinstance(major_tick_intervals, float) or not isinstance(
            minor_tick_intervals, float
        ):
            raise TypeError(
                "`major_tick_intervals` and `minor_tick_intervals` should be "
                "floats."
            )

        return connections, f1, f2, f1_idcs, f2_idcs

    def _create_plots(
        self, connections: list[int], n_rows: int, n_cols: int
    ) -> tuple[list[Figure], list[np.ndarray]]:
        """Create figures and subplots to fill with results.

        Returns
        -------
        figures : list of matplotlib Figure
            Figures for the results in a list of length
            ``ceil(n_cons / (n_rows * n_cols))``.

        axes : list of numpy.ndarray of matplotlib pyplot Axes
            Subplot axes for the results in a list of length
            ``ceil(n_cons / (n_rows * n_cols))`` where each entry is a 1D
            ``numpy.ndarray`` of length ``(n_rows * n_cols)``.
        """
        figures = []
        axes = []

        plot_n = 0
        for con_i in range(len(connections)):
            if con_i == plot_n:
                fig, axs = plt.subplots(n_rows, n_cols, layout="constrained")
                figures.append(fig)
                if n_rows * n_cols > 1:
                    axs = np.ravel(axs)
                else:
                    axs = np.array([axs])
                axes.append(axs)
                plot_n += n_rows * n_cols
            if plot_n >= len(connections):
                break

        return figures, axes

    def _plot_results(
        self,
        figures: list[Figure],
        axes: list[np.ndarray],
        connections: list[int],
        f1: np.ndarray,
        f2: np.ndarray,
        f1_idcs: list[int],
        f2_idcs: list[int],
        n_rows: int,
        n_cols: int,
        major_tick_intervals: float,
        minor_tick_intervals: float,
    ) -> None:
        """Plot results on the relevant figures/subplots."""
        fig_i = 0
        plot_n = 0
        fig_plot_n = 0
        for _ in range(n_rows):
            for _ in range(n_cols):
                con_i = connections[plot_n]
                axis = axes[fig_i][fig_plot_n]

                mesh = axis.pcolormesh(
                    f1, f2, self._data[con_i][np.ix_(f1_idcs, f2_idcs)].T
                )

                plt.colorbar(
                    mesh, ax=axis, label="Coupling (A.U.)", shrink=0.3
                )

                axis.set_aspect("equal")
                self._set_axis_ticks(
                    axis, f1, f2, major_tick_intervals, minor_tick_intervals
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

                axis.set_title(
                    f"Seed: {self._seeds[con_i]} | Target: "
                    f"{self._targets[con_i]}"
                )

                plot_n += 1
                fig_plot_n += 1
                if fig_plot_n >= n_rows * n_cols:
                    figures[fig_i].suptitle(self.name)
                    fig_plot_n = 0
                    fig_i += 1

    def _set_axis_ticks(
        self,
        axis: plt.Axes,
        f1: np.ndarray,
        f2: np.ndarray,
        major_tick_intervals: float,
        minor_tick_intervals: float,
    ) -> None:
        """Set major and minor tick intervals of x- and y-axes."""
        n_major_xticks = len(np.arange(f1[0], f1[-1], major_tick_intervals))
        n_major_yticks = len(np.arange(f2[0], f2[-1], major_tick_intervals))

        # extra tick necessary if 0 Hz plotted
        if f1[0] == 0.0:
            n_major_xticks += 1
        if f2[0] == 0.0:
            n_major_yticks += 1

        # MaxNLocator only cares about tens (e.g. 10 and 19 have same result)
        n_minor_xticks = (
            np.ceil(len(np.arange(f1[0], f1[-1], minor_tick_intervals)) / 10)
            * 10
        )
        n_minor_yticks = (
            np.ceil(len(np.arange(f2[0], f2[-1], minor_tick_intervals)) / 10)
            * 10
        )

        axis.xaxis.set_major_locator(plt.MaxNLocator(n_major_xticks))
        axis.xaxis.set_minor_locator(plt.MaxNLocator(n_minor_xticks))
        axis.yaxis.set_major_locator(plt.MaxNLocator(n_major_yticks))
        axis.yaxis.set_minor_locator(plt.MaxNLocator(n_minor_yticks))


class ResultsTDE:
    """Class for storing time delay estimation (TDE) results.

    Parameters
    ----------
    data : numpy.ndarray of float
        2D array of results to store with shape `[connections x times]`.

    indices : tuple of numpy.ndarray of int
        Indices of the channels for each connection of :attr:`data`. Should
        contain two 1D arrays of equal length for the seed and target indices,
        respectively.

    times : numpy.ndarray of float
        1D array of timepoints in :attr:`data`.

    name : str
        Name of the results being stored.

    Attributes
    ----------
    name : str
        Name of the results.

    indices : tuple of numpy.ndarray of int
        Indices of the channels for each connection of the results. Contains
        two 1D arrays of equal length for the seed and target indices,
        respectively.

    n_cons : str
        Number of connections in the results.

    times : numpy.ndarray of float
        1D array of timepoints in ``data``.
    """

    _data = None

    indices = None
    _seeds = None
    _targets = None
    n_cons = None
    _n_chans = None

    times = None

    name = None

    def __init__(
        self,
        data: np.ndarray,
        indices: tuple[np.ndarray],
        times: np.ndarray,
        name: str,
    ) -> None:
        self._sort_init_inputs(data, indices, times, name)

    def _sort_init_inputs(
        self,
        data: np.ndarray,
        indices: tuple[np.ndarray],
        times: np.ndarray,
        name: str,
    ) -> None:
        """Sort inputs to the object."""
        if not isinstance(data, np.ndarray):
            raise TypeError("`data` must be a NumPy array.")
        if data.ndim != 2:
            raise ValueError("`data` must be a 2D array.")
        self._data = data.copy()

        if not isinstance(indices, tuple):
            raise TypeError("`indices` must be a tuple.")
        if len(indices) != 2:
            raise ValueError("`indices` must have a length of 2.")
        if not isinstance(indices[0], np.ndarray) or not isinstance(
            indices[1], np.ndarray
        ):
            raise TypeError("Entries of `indices` must be NumPy arrays.")
        if indices[0].ndim != 1 or indices[1].ndim != 1:
            raise ValueError("Entries of `indices` must be 1D arrays.")
        if len(indices[0]) != len(indices[1]):
            raise ValueError("Entries of `indices` must have the same length.")
        self.indices = copy.copy(indices)
        self._seeds = indices[0].copy()
        self._targets = indices[1].copy()
        self.n_cons = len(indices[0])
        self._n_chans = len(np.unique([*self._seeds, *self._targets]))

        if not isinstance(times, np.ndarray):
            raise TypeError("`times` must be NumPy arrays.")
        if times.ndim != 1:
            raise ValueError("`times` must be 1D arrays.")
        self.times = times.copy()

        if data.shape != (len(indices[0]), len(times)):
            raise ValueError("`data` must have shape [connections x times].")

        if not isinstance(name, str):
            raise TypeError("`name` must be a string.")
        self.name = copy.copy(name)

    def __repr__(self) -> str:
        """Return printable represenation of the object."""
        return repr(
            f"<Result: {self.name} | [{self.n_cons} connections x "
            f"{len(self.times)} times]>"
        )


def compute_fft(
    data: np.ndarray,
    sfreq: int,
    n_jobs: int = 1,
    verbose: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the FFT on real-valued data.

    As the data is assumed to be real-valued, only those values corresponding
    to the positive frequencies are returned.

    Parameters
    ----------
    data : numpy.ndarray of float
        3D array of real-valued data to compute the FFT on, with shape `[epochs
        x channels x times]`.

    sfreq : int
        Sampling frequency of the data in Hz.

    n_jobs : int (default ``1``)
        Number of jobs to run in parallel.

    verbose : bool (default ``True``)
        Whether or not to report the status of the processing.

    Returns
    -------
    fft : numpy.ndarray of float
        3D array of FFT coefficients of the data with shape [epochs x channels
        x positive frequencies].

    freqs : numpy.ndarray of float
        1D array of the frequencies in ``fft``.
    """
    if not isinstance(data, np.ndarray):
        raise TypeError("`data` must be a NumPy NDArray.")
    if data.ndim != 3:
        raise ValueError("`data` must be a 3D array.")

    if not isinstance(n_jobs, int):
        raise TypeError("`n_jobs` must be an integer.")
    if n_jobs < 1:
        raise ValueError("`n_jobs` must be >= 1.")

    if not isinstance(sfreq, int):
        if isinstance(sfreq, float):
            if verbose:
                warn(
                    "`sfreq` is a float. Converting it to an int.", UserWarning
                )
        else:
            raise TypeError("`sfreq` must be an int.")

    if verbose and not np.isreal(data).all():
        warn("`data` is expected to be real-valued.", UserWarning)

    if verbose:
        print("Computing FFT on the data...")

    freqs = np.linspace(0.0, sfreq / 2.0, int(sfreq) + 1)

    window = np.hanning(data.shape[2])

    args = [
        {"a": sp.signal.detrend(chan_data) * window}
        for chan_data in data.transpose(1, 0, 2)
    ]

    fft = np.array(
        pqdm(
            args,
            np.fft.fft,
            n_jobs,
            argument_type="kwargs",
            desc="Processing channels...",
            disable=not verbose,
        )
    ).transpose(1, 0, 2)

    if verbose:
        print("    [FFT computation finished]\n")

    return fft[..., : len(freqs)], freqs


@njit
def fast_find_first(vector: np.ndarray, value: float | int) -> int:
    """Quickly find the first index of a value in a 1D array using Numba.

    Parameters
    ----------
    vector : numpy.ndarray of float or int
        1D array to find ``value`` in.

    value : float | int
        value to find in ``vector``.

    Returns
    -------
    index : int
        First index of ``value`` in ``vector``.

    Notes
    -----
    Does not perform checks in inputs for speed.
    """
    for idx, val in enumerate(vector):
        if val == value:
            return idx
    raise ValueError("`value` is not present in `vector`.")


def compute_rank(data: np.ndarray, sv_tol: float = 1e-5) -> int:
    """Compute the min. rank of data over epochs from non-zero singular values.

    Parameters
    ----------
    data : numpy.ndarray, shape [epochs x channels x timepoints]
        Data to find the rank of.

    sv_tol : float (default 1e-5)
        Tolerance to use to define non-zero singular values based on the
        largest singular value.

    Returns
    -------
    rank : int
        Minimum rank of ``data`` over epochs.
    """
    if not isinstance(data, np.ndarray):
        raise TypeError("`data` must be a NumPy array.")
    if data.ndim != 3:
        raise ValueError("`data` must be a 3D array.")

    if not isinstance(sv_tol, float) and not isinstance(sv_tol, int):
        raise TypeError("`sv_tol` must be a float or an int.")

    singular_vals = np.linalg.svd(data, compute_uv=False).min(axis=0)

    return np.count_nonzero(singular_vals > singular_vals[0] * sv_tol)


def _generate_data(
    n_epochs: int, n_chans: int, n_times: int, seed: int = 44
) -> np.ndarray:
    """Generate random data of the specified shape."""
    random = np.random.RandomState(seed)
    return random.rand(n_epochs, n_chans, n_times)
