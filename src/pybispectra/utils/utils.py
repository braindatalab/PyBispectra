"""Helper tools for processing CFC results."""

import copy
from warnings import warn

import numpy as np
import scipy as sp
from matplotlib.figure import Figure
from matplotlib import pyplot as plt
from pqdm.processes import pqdm
from numba import njit


class Results:
    """Class for storing results.

    PARAMETERS
    ----------
    data : NumPy ndarray
    -   3D array of results to store with shape [connections x f2 x f1].

    indices : tuple of NumPy ndarray
    -   Indices of the channels for each connection of `data`. Should contain 2
        1D arrays of equal length for the seed and target indices,
        respectively.

    f1 : NumPy ndarray
    -   1D array of low frequencies in `data`.

    f2 : NumPy ndarray
    -   1D array of high frequencies in `data`.

    name : str
    -   Name of the results being stored.

    METHODS
    -------
    get_results
    -   Return a copy of results as arrays.

    plot
    -   Plots the results.

    ATTRIBUTES
    ----------
    name : str
    -   Name of the results.

    indices : tuple of NumPy ndarray
    -   Indices of the channels for each connection of the results. Contains 2
        1D arrays of equal length for the seed and target indices,
        respectively.

    n_cons : str
    -   Number of connections in the results.

    f1 : NumPy ndarray
    -   1D array of low frequencies in the results.

    f2 : NumPy ndarray
    -   1D array of high frequencies in the results.
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
        self._n_chans = np.unique([*self._seeds, *self._targets])

        if not isinstance(f1, np.ndarray) or not isinstance(f2, np.ndarray):
            raise TypeError("`f1` and `f2` must be NumPy arrays.")
        if f1.ndim != 1 or f2.ndim != 1:
            raise TypeError("`f1` and `f2` must be 1D arrays.")
        self.f1 = f1.copy()
        self.f2 = f2.copy()

        if data.shape != (len(indices[0]), len(f2), len(f1)):
            raise ValueError("`data` must have shape [connections x f2 x f1].")

        if not isinstance(name, str):
            raise TypeError("`name` must be a string.")
        self.name = copy.copy(name)

    def get_results(
        self, form: str = "raveled"
    ) -> np.ndarray | tuple[np.ndarray, tuple[np.ndarray]]:
        """Return a copy of the results as arrays.

        PARAMETERS
        ----------
        form : str; default "raveled"
        -   How the results should be returned: "raveled" - results have shape
            [connections x f2 x f1]; "compact" - results have shape [seeds x
            targets x f2 x f1].

        RETURNS
        -------
        results : NumPy ndarray
        -   Spectral coupling results.

        indices : tuple of NumPy ndarray
        -   Channel indices of the seeds and targets. Only returned if `form`
            is "compact".
        """
        accepted_forms = ["raveled", "compact"]
        if form not in accepted_forms:
            raise ValueError("`form` is not recognised.")

        if form == "raveled":
            return self._data.copy()
        return self._get_compact_results()

    def _get_compact_results(self) -> tuple[np.ndarray, tuple[np.ndarray]]:
        """Return a compacted form of the results.

        RETURNS
        -------
        compact_results : NumPy ndarray
        -   Results with shape [seeds x targets x f2 x f1].

        indices : tuple[NumPy ndarray]
        -   Channel indices of `compact_results`, for the seeds and targets,
            respectively.
        """
        fill = np.full((len(self.f2), len(self.f1)), fill_value=np.nan)
        compact_results = np.full(
            (self._n_chans, self._n_chans, len(self.f2), len(self.f1)),
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
        show: bool = True,
    ) -> None:
        """Plot the results.

        PARAMETERS
        ----------
        connections : list of int | None; default None
        -   Indices of connections to plot. If None, plot all connections.

        f1 : NumPy ndarray | None; default None
        -   Low frequencies of the results to plot. If None, plot all low
            frequencies.

        f2 : NumPy ndarray | None; default None
        -   High frequencies of the results to plot. If None, plot all high
            frequencies.

        n_rows : int; default 1
        -   Number of rows of subplots per figure.

        n_cols : int; default 1
        -   Number of columns of subplots per figure.

        show : bool; default True
        -   Whether or not to show the plotted results.

        RETURNS
        -------
        figures : list of matplotlib Figure
        -   Figures of the results in a list of length
            ceil(n_cons / (n_rows * n_cols)).

        axes : list of matplotlib pyplot Axes
        -   Subplot axes for the results in a list of length
            ceil(n_cons / (n_rows * n_cols)) where each entry is a 1D NumPy
            array of length (n_rows * n_cols).

        NOTES
        -----
        -   `n_rows` and `n_cols` of 1 will plot the results for each
            connection on a new figure.
        """
        connections, f1, f2, f1_idcs, f2_idcs = self._sort_plot_inputs(
            connections, f1, f2, n_rows, n_cols
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
    ) -> tuple[list[int], np.ndarray, np.ndarray, list[int], list[int]]:
        """Sort the plotting inputs.

        RETURNS
        -------
        connections : list of int
        -   Indices of connections to plot.

        f1 : NumPy ndarray
        -   Low frequencies of the results to plot.

        f2 : NumPy ndarray
        -   High frequencies of the results to plot.

        f1_idcs : list of int
        -   Indices of `f1` in the results.

        f2_idcs : list of int
        -   Indices of `f2` in the results.
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

        return connections, f1, f2, f1_idcs, f2_idcs

    def _create_plots(
        self, connections: list[int], n_rows: int, n_cols: int
    ) -> tuple[list[Figure], list[plt.Axes]]:
        """Create figures and subplots to fill with results.

        RETURNS
        -------
        figures : list of matplotlib Figure
        -   Figures for the results in a list of length
            ceil(n_cons / (n_rows * n_cols)).

        axes : list of NumPy array of matplotlib pyplot Axes
        -   Subplot axes for the results in a list of length
            ceil(n_cons / (n_rows * n_cols)) where each entry is a 1D NumPy
            array of length (n_rows * n_cols).
        """
        figures = []
        axes = []

        plot_n = 0
        for con_i in range(len(connections)):
            if con_i == plot_n:
                fig, axs = plt.subplots(n_rows, n_cols)
                figures.append(fig)
                axes.append(np.flatten(axs))
                plot_n += n_rows * n_cols
            if plot_n >= len(connections):
                break

        return figures, axes

    def _plot_results(
        self,
        figures: list[Figure],
        axes: list[np.ndarray[plt.Axes]],
        connections: list[int],
        f1: np.ndarray,
        f2: np.ndarray,
        f1_idcs: list[int],
        f2_idcs: list[int],
        n_rows: int,
        n_cols: int,
    ) -> None:
        """Plot results on the relevant figures/subplots."""
        fig_i = 0
        plot_n = 0
        fig_plot_n = 0
        for row_i in range(n_rows):
            for col_i in range(n_cols):
                con_i = connections[plot_n]
                axis = axes[fig_i][row_i + col_i]

                axis.imshow(self._data[np.ix_(con_i, f2_idcs, f1_idcs)])

                axis.grid(
                    which="major",
                    axis="both",
                    linestyle="-",
                    color=[0.3, 0.3, 0.3],
                    alpha=0.3,
                )
                axis.set_xlabel("$F_2$ (Hz)")
                axis.set_xticklabels(f2)
                axis.set_ylabel("$F_1$ (Hz)")
                axis.set_yticklabels(f1)

                axis.set_title(
                    f"Seed: {self._seeds[con_i]} | Target: "
                    f"{self._targets[con_i]}"
                )

                plot_n += 1
                fig_plot_n += 1
                if fig_plot_n > n_rows * n_cols:
                    figures[fig_i].title(self.name)
                    fig_plot_n = 0
                    fig_i += 1


def compute_fft(
    data: np.ndarray, sfreq: float, n_jobs: int = 1, verbose: bool = True
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the FFT on real-valued data.

    As the data is assumed to be real-valued, only those values corresponding
    to the positive frequencies are returned.

    PARAMETERS
    ----------
    data : NumPy ndarray
    -   3D array of real-valued data to compute the FFT on, with shape [epochs
        x channels x times].

    sfreq : float
    -   Sampling frequency of the data in Hz.

    n_jobs : int; default 1
    -   Number of jobs to run in parallel

    verbose : bool; default True
    -   Whether or not to report the status of the processing.

    RETURNS
    -------
    fft : NumPy ndarray
    -   3D array of FFT coefficients of the data with shape [epochs x channels
        x positive frequencies].

    freqs : NumPy ndarray
    -   1D array of the frequencies in `fft`.

    RAISES
    ------
    ValueError
    -   Raised if `data` is not a NumPy ndarray or does not have 3 dimensions.
    """
    if not isinstance(data, np.ndarray):
        raise TypeError("`data` must be a NumPy array.")
    if data.ndim != 3:
        raise ValueError("`data` must be a 3D array.")

    if not isinstance(n_jobs, int):
        raise TypeError("`n_jobs` must be an integer.")
    if n_jobs < 1:
        raise ValueError("`n_jobs` must be >= 1.")

    if not np.isreal(data).all():
        warn("`data` is expected to be real-valued.", UserWarning)

    if verbose:
        print("Computing FFT on the data...")

    freqs = np.linspace(0, sfreq / 2, sfreq + 1)

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

    return fft[..., 1 : len(freqs)], freqs[1:]  # ignore zero freq.


@njit
def fast_find_first(vector: np.ndarray, value: float) -> int:
    """Quickly find the first index of a value in a 1D array using Numba.

    PARAMETERS
    ----------
    vector : NumPy ndarray
    -   1D array to find `value` in.

    value : float
    -   value to find in `vector`.

    RETURNS
    -------
    index : int
    -   First index of `value` in `vector`.

    NOTES
    -----
    -   Does not check if `vector` is a 1D NumPy array or if `value` is a
        single value for speed.
    """
    for idx, val in enumerate(vector):
        if val == value:
            return idx
    raise ValueError("`value` is not present in `vector`.")


def generate_data(
    n_epochs: int, n_chans: int, n_times: int, seed: int = 44
) -> np.ndarray:
    """Generate random data of the specified shape."""
    random = np.random.RandomState(seed)
    return random.rand((n_epochs, n_chans, n_times))
