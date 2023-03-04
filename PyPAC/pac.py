"""Tools for handling PAC analysis."""

import copy
from warnings import warn

import numpy as np
from numba import njit
from pqdm.processes import pqdm

from process import Process
from utils import fast_find_first


np.seterr(divide="ignore", invalid="ignore")  # no warning for NaN division


class PAC(Process):
    """Class for computing phase-amplitude (PAC) coupling using bispectra.

    PARAMETERS
    ----------
    data : NumPy ndarray
    -   3D array of FFT coefficients with shape [epochs x channels x
        frequencies].

    freqs : NumPy ndarray
    -   1D array of the frequencies in `data`.

    verbose : bool; default True
    -   Whether or not to report the progress of the processing.

    METHODS
    -------
    compute
    -   Compute PAC, averaged over epochs.

    get_results
    -   Return a copy of the results.

    copy
    -   Return a copy of the object.

    ATTRIBUTES
    ----------
    data : NumPy ndarray
    -   FFT coefficients with shape [epochs x channels x frequencies].

    freqs : NumPy ndarray
    -   1D array of the frequencies in `data`.

    indices : tuple of NumPy ndarray
    -   2 arrays containing the seed and target indices (respectively) most
        recently used with `compute`.

    f1 : NumPy ndarray
    -   1D array of low frequencies most recently used with `compute`.

    f2 : NumPy ndarray
    -   1D array of high frequencies most recently used with `compute`.

    verbose : bool
    -   Whether or not to report the progress of the processing.
    """

    _return_nosym = False
    _return_antisym = False
    _return_nonorm = False
    _return_threenorm = False

    _bispectra = None
    _bicoherence = None

    _pac_nosym_nonorm = None
    _pac_nosym_threenorm = None
    _pac_antisym_nonorm = None
    _pac_antisym_threenorm = None

    def compute(
        self,
        indices: tuple[np.ndarray] | None = None,
        f1: np.ndarray | None = None,
        f2: np.ndarray | None = None,
        symmetrise: str | list[str] = ["none", "antisym"],
        normalise: str | list[str] = ["none", "threenorm"],
        n_jobs: int = 1,
    ) -> None:
        """Compute PAC, averaged over epochs.

        PARAMETERS
        ----------
        indices: tuple of NumPy ndarray of int | None; default None
        -   Indices of the channels to compute PAC between. Should contain 2
            1D arrays of equal length for the seed and target indices,
            respectively. If None, coupling between all channels is computed.

        f1 : numpy ndarray | None; default None
        -   A 1D array of the lower frequencies to compute PAC on. If None, all
            frequencies are used.

        f2 : numpy ndarray | None; default None
        -   A 1D array of the higher frequencies to compute PAC on. If None,
            all frequencies are used.

        symmetrise : str | list of str; default ["none", "antisym"]
        -   Symmetrisation to perform when computing PAC. If "none", no
            symmetrisation is performed. If "antisym", antisymmetrisation is
            performed.

        normalise : str | list of str; default ["none", "threenorm"]
        -   Normalisation to perform when computing PAC. If "none", no
            normalisation is performed. If "threenorm", the bispectra is
            normalised to the bicoherence using a threenorm.

        n_jobs : int; default 1
        -   The number of jobs to run in parallel.

        NOTES
        -----
        -   PAC is computed between all values of `f1` and `f2`. If any value
            of `f1` is higher than `f2`, a NaN value is returned.
        """
        self._reset_attrs()

        self._sort_metrics(symmetrise, normalise)
        self._sort_indices(indices)
        self._sort_freqs(f1, f2)
        self._sort_parallelisation(n_jobs)

        if self.verbose:
            print("Computing PAC...\n")

        self._compute_bispectra()
        if self._return_threenorm:
            self._compute_bicoherence()
        self._compute_pac()

        if self.verbose:
            print("    [PAC computation finished]\n")

    def _reset_attrs(self) -> None:
        """Reset attrs. of the object to prevent interference."""
        super()._reset_attrs()

        self._return_nosym = False
        self._return_antisym = False
        self._return_nonorm = False
        self._return_threenorm = False

        self._bispectra = None
        self._bicoherence = None

        self._pac_nosym_nonorm = None
        self._pac_nosym_threenorm = None
        self._pac_antisym_nonorm = None
        self._pac_antisym_threenorm = None

    def _sort_indices(self, indices: np.ndarray) -> None:
        """Sort seed-target indices inputs."""
        super()._sort_indices(indices)

        if (
            any(
                seed == target
                for seed, target in zip(self._seeds, self._targets)
            )
            and self._return_antisym
        ):
            warn(
                "The seed and target for at least one connection is the same "
                "channel. The corresponding antisymmetrised result(s) will be "
                "zero.",
                UserWarning,
            )

    def _sort_freqs(self, f1: np.ndarray, f2: np.ndarray) -> None:
        """Sort frequency inputs."""
        super()._sort_freqs(f1, f2)

        if self.verbose:
            if any(
                hfreq + lfreq not in self.freqs
                for hfreq in self.f2
                for lfreq in self.f1
            ):
                warn(
                    "At least one value of `f2` + `f1` is not present in the "
                    "frequencies. The corresponding result(s) will have a "
                    "value of NaN.",
                    UserWarning,
                )

    def _sort_metrics(
        self, symmetrise: str | list[str], normalise: str | list[str]
    ) -> None:
        """Sort inputs for the form of results being requested."""
        if not isinstance(symmetrise, str) and not isinstance(
            symmetrise, list
        ):
            raise TypeError(
                "`symmetrise` must be a list of strings or a string."
            )
        if not isinstance(normalise, str) and not isinstance(normalise, list):
            raise TypeError(
                "`normalise` must be a list of strings or a string."
            )

        if isinstance(symmetrise, str):
            symmetrise = [copy.copy(symmetrise)]
        if isinstance(normalise, str):
            normalise = [copy.copy(normalise)]

        supported_sym = ["none", "antisym"]
        if any(entry not in supported_sym for entry in symmetrise):
            raise ValueError("The value of `symmetrise` is not recognised.")
        supported_norm = ["none", "threenorm"]
        if any(entry not in supported_norm for entry in normalise):
            raise ValueError("The value of `normalise` is not recognised.")

        if "none" in symmetrise:
            self._return_nosym = True
        if "antisym" in symmetrise:
            self._return_antisym = True

        if "none" in normalise:
            self._return_nonorm = True
        if "threenorm" in normalise:
            self._return_threenorm = True

    def _compute_bispectra(self) -> None:
        """Compute bispectra between f1s and f2s of channels in `indices`."""
        if self.verbose:
            print("    Computing bispectra...")

        args = [
            {
                "data": self.data[:, (seed, target)],
                "freqs": self.freqs,
                "f1s": self.f1,
                "f2s": self.f2,
            }
            for seed, target in zip(self._seeds, self._targets)
        ]

        # have to average complex value outside of Numba-compiled function
        self._bispectra = (
            np.array(
                pqdm(
                    args,
                    _compute_bispectra,
                    self._n_jobs,
                    argument_type="kwargs",
                    desc="Processing connections...",
                    disable=not self.verbose,
                )
            )
            .mean(axis=2)
            .transpose(1, 0, 2, 3)
        )

        if self.verbose:
            print("        [Bispectra computation finished]\n")

    def _compute_bicoherence(self) -> None:
        """Compute bicoherence from the bispectra using the threenorm."""
        if self.verbose:
            print("    Computing bicoherence...")

        args = [
            {
                "data": self.data[:, (seed, target)],
                "freqs": self.freqs,
                "f1s": self.f1,
                "f2s": self.f2,
            }
            for seed, target in zip(self._seeds, self._targets)
        ]

        threenorm = np.array(
            pqdm(
                args,
                _compute_threenorm,
                self._n_jobs,
                argument_type="kwargs",
                desc="Processing connections...",
                disable=not self.verbose,
            )
        )

        self._bicoherence = self._bispectra / threenorm

        if self.verbose:
            print("        [Bicoherence computation finished]\n")

    def _compute_pac(self) -> None:
        """Compute PAC results from bispectra/bicoherence."""
        if self._return_nonorm:
            if self._return_nosym:
                self._pac_nosym_nonorm = np.abs(self._bispectra[0])
            if self._return_antisym:
                self._pac_antisym_nonorm = np.abs(
                    self._bispectra[0] - self._bispectra[1]
                )

        if self._return_threenorm:
            if self._return_nosym:
                self._pac_nosym_threenorm = np.abs(self._bicoherence[0])
            if self._return_antisym:
                self._pac_antisym_threenorm = np.abs(
                    self._bicoherence[0] - self._bicoherence[1]
                )

    def get_results(
        self, form: str = "raveled"
    ) -> tuple[np.ndarray, str] | tuple[tuple[np.ndarray], tuple[str]] | tuple[
        np.ndarray, str, tuple[np.ndarray]
    ] | tuple[tuple[np.ndarray], tuple[str], tuple[np.ndarray]]:
        """Return a copy of the results.

        PARAMETERS
        ----------
        form : str; default "raveled"
        -   How the results should be returned: "raveled" - results have shape
            [connections x f2 x f1]; "compact" - results have shape [seeds x
            targets x f2 x f1].

        RETURNS
        -------
        results : NumPy ndarray | tuple of NumPy ndarray
        -   PAC results. If multiple types of PAC computed, `results` is a
            tuple of arrays.

        result_types : str | tuple of str
        -   Types of PAC results according to the entries of `results`. If
            multiple types of PAC computed, `result_types` if a tuple of
            strings.

        indices : tuple of NumPy ndarray
        -   Channel indices of the seeds and targets. Only returned if `form`
            is "compact".
        """
        accepted_forms = ["raveled", "compact"]
        if form not in accepted_forms:
            raise ValueError("`form` is not recognised.")

        results, result_types = self._get_results()

        if form == "compact":
            results = [self._get_compact_results(result) for result in results]
            indices = results[0][1]
            results = tuple(result[0] for result in results)
            if len(results) == 1:
                results = results[0]
                result_types = result_types[0]
            return results, result_types, indices

        # raveled results
        if len(results) == 1:
            results = results[0]
            result_types = result_types[0]
        return results, result_types

    def _get_results(self) -> tuple[tuple[np.ndarray], tuple[str]]:
        """Return a copy of the PAC results and their types."""
        results = []
        result_types = []
        if self._pac_nosym_nonorm is not None:
            results.append(self._pac_nosym_nonorm.copy())
            result_types.append("standard_bispectra_pac")

        if self._pac_nosym_threenorm is not None:
            results.append(self._pac_nosym_threenorm.copy())
            result_types.append("standard_bicoherence_pac")

        if self._pac_antisym_nonorm is not None:
            results.append(self._pac_antisym_nonorm.copy())
            result_types.append("antisymmetrised_bispectra_pac")

        if self._pac_antisym_threenorm is not None:
            results.append(self._pac_antisym_threenorm.copy())
            result_types.append("antisymmetrised_bicoherence_pac")

        return tuple(results), tuple(result_types)


@njit
def _compute_bispectra(
    data: np.ndarray, freqs: np.ndarray, f1s: np.ndarray, f2s: np.ndarray
) -> np.ndarray:
    """Compute bispectra for a single connection.

    PARAMETERS
    ----------
    data : NumPy ndarray
    -   3D array of FFT coefficients with shape [epochs x 2 x frequencies],
        where the second dimension contains the data for the seed and target
        channel of a single connection, respectively.

    freqs : NumPy ndarray
    -   1D array of frequencies in `data`.

    f1s : NumPy ndarray
    -   1D array of low frequencies to compute bispectra for.

    f2s : NumPy ndarray
    -   1D array of high frequencies to compute bispectra for.

    RETURNS
    -------
    results : NumPy ndarray
    -   4D array containing the bispectra of a single connection with shape [2
        x epochs x f2 x f1], where the first dimension corresponds to the
        standard bispectra (B_kmm) and symmetric bispectra (B_mkm),
        respectively (where k is the seed and m is the target).

    NOTES
    -----
    -   Averaging across epochs is not performed here as `np.mean` of complex
        numbers if not supported when compiling using Numba.
    """
    results = np.full(
        (2, data.shape[0], f2s.shape[0], f1s.shape[0]),
        fill_value=np.nan,
        dtype=np.complex128,
    )
    for f1_i, f1 in enumerate(f1s):
        for f2_i, f2 in enumerate(f2s):
            if f1 < f2 and (f2 + f1) in freqs:
                for epoch_i, epoch_data in enumerate(data):
                    f1_loc = fast_find_first(freqs, f1)
                    f2_loc = fast_find_first(freqs, f1)

                    # B_kmm
                    fft_f1 = epoch_data[0, f1_loc]
                    fft_f2 = epoch_data[1, f2_loc]
                    fft_fdiff = epoch_data[1, fast_find_first(freqs, f2 + f1)]
                    results[0, epoch_i, f2_i, f1_i] = fft_f1 * (
                        fft_fdiff * np.conjugate(fft_f2)
                    )

                    # B_mkm
                    fft_f1 = epoch_data[1, f1_loc]
                    fft_f2 = epoch_data[0, f2_loc]
                    results[1, epoch_i, f2_i, f1_i] = fft_f1 * (
                        fft_fdiff * np.conjugate(fft_f2)
                    )

    return results


@njit
def _compute_threenorm(
    data: np.ndarray, freqs: np.ndarray, f1s: np.ndarray, f2s: np.ndarray
) -> np.ndarray:
    """Compute threenorm for a single connection across epochs.

    PARAMETERS
    ----------
    data : NumPy ndarray
    -   3D array of FFT coefficients with shape [epochs x 2 x frequencies],
        where the second dimension contains the data for the seed and target
        channel of a single connection, respectively.

    freqs : NumPy ndarray
    -   1D array of frequencies in `data`.

    f1s : NumPy ndarray
    -   1D array of low frequencies to compute threenorm for.

    f2s : NumPy ndarray
    -   1D array of high frequencies to compute threenorm for.

    RETURNS
    -------
    results : NumPy ndarray
    -   2D array containing the threenorm of a single connection averaged
        across epochs, with shape [f2 x f1].
    """
    results = np.full(
        (f2s.shape[0], f1s.shape[0]), fill_value=np.nan, dtype=np.float64
    )
    for f1_i, f1 in enumerate(f1s):
        for f2_i, f2 in enumerate(f2s):
            if f1 < f2 and (f2 + f1) in freqs:
                fft_f1 = data[:, 0, fast_find_first(freqs, f1)]
                fft_f2 = data[:, 1, fast_find_first(freqs, f2)]
                fft_fdiff = data[:, 1, fast_find_first(freqs, f2 + f1)]
                results[f2_i, f1_i] = (
                    (np.abs(fft_f1) ** 3).mean()
                    * (np.abs(fft_f2) ** 3).mean()
                    * (np.abs(fft_fdiff) ** 3).mean()
                ) ** 1 / 3

    return results
