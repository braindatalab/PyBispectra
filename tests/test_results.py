"""Tests for results classes (plotting tested separately)."""

import numpy as np
import pytest

from pybispectra.utils import ResultsCFC, ResultsGeneral, ResultsTDE, ResultsWaveShape
from pybispectra.utils._utils import _generate_data


def test_results_cfc_error_catch() -> None:
    """Test `ResultsCFC` catches errors."""
    n_cons = 9
    n_f1 = 50
    n_f2 = 50
    data = _generate_data(n_cons, n_f1, n_f2)
    f1s = np.arange(n_f1)
    f2s = np.arange(n_f2)
    n_unique_chans = 3
    indices = (
        tuple(np.repeat(np.arange(n_unique_chans), n_unique_chans).tolist()),
        tuple(np.tile(np.arange(n_unique_chans), n_unique_chans).tolist()),
    )

    with pytest.raises(TypeError, match="`data` must be a NumPy array."):
        ResultsCFC(data=data.tolist(), indices=indices, f1s=f1s, f2s=f2s)
    with pytest.raises(ValueError, match="`data` must be a 3D array."):
        ResultsCFC(data=data[..., 0], indices=indices, f1s=f1s, f2s=f2s)

    with pytest.raises(TypeError, match="`indices` must be a tuple."):
        ResultsCFC(data=data, indices=list(indices), f1s=f1s, f2s=f2s)
    with pytest.raises(ValueError, match="`indices` must have length of 2."):
        ResultsCFC(
            data=data, indices=(indices[0], indices[0], indices[1]), f1s=f1s, f2s=f2s
        )
    with pytest.raises(TypeError, match="Entries of `indices` must be tuples."):
        ResultsCFC(data=data, indices=(0, 1), f1s=f1s, f2s=f2s)
    with pytest.raises(
        TypeError, match="Entries for seeds and targets in `indices` must be ints."
    ):
        ResultsCFC(data=data, indices=((0.5,), (1.5,)), f1s=f1s, f2s=f2s)
    with pytest.raises(
        ValueError, match="Entries of `indices` must have equal length."
    ):
        ResultsCFC(
            data=data,
            indices=(indices[0], tuple(np.concatenate((indices[1], [1])).tolist())),
            f1s=f1s,
            f2s=f2s,
        )
    with pytest.raises(
        ValueError,
        match=("Entries for seeds and targets in `indices` must be >= 0."),
    ):
        ResultsCFC(data=data, indices=((0,), (-1,)), f1s=f1s, f2s=f2s)

    with pytest.raises(TypeError, match="`f1s` and `f2s` must be NumPy arrays."):
        ResultsCFC(data=data, indices=indices, f1s=f1s.tolist(), f2s=f2s)
    with pytest.raises(TypeError, match="`f1s` and `f2s` must be NumPy arrays."):
        ResultsCFC(data=data, indices=indices, f1s=f1s, f2s=f2s.tolist())
    with pytest.raises(ValueError, match="`f1s` and `f2s` must be 1D arrays."):
        ResultsCFC(data=data, indices=indices, f1s=np.vstack((f1s, f1s)), f2s=f2s)
    with pytest.raises(ValueError, match="`f1s` and `f2s` must be 1D arrays."):
        ResultsCFC(data=data, indices=indices, f1s=f1s, f2s=np.vstack((f2s, f2s)))

    with pytest.raises(
        ValueError, match=r"`data` must have shape \[nodes, f1s, f2s\]."
    ):
        ResultsCFC(data=data[1:, :, :], indices=indices, f1s=f1s, f2s=f2s)
    with pytest.raises(
        ValueError, match=r"`data` must have shape \[nodes, f1s, f2s\]."
    ):
        ResultsCFC(data=data, indices=indices, f1s=f1s[1:], f2s=f2s)
    with pytest.raises(
        ValueError, match=r"`data` must have shape \[nodes, f1s, f2s\]."
    ):
        ResultsCFC(data=data, indices=indices, f1s=f1s, f2s=f2s[1:])

    with pytest.raises(TypeError, match="`name` must be a string."):
        ResultsCFC(data=data, indices=indices, f1s=f1s, f2s=f2s, name=1)

    results = ResultsCFC(data=data, indices=indices, f1s=f1s, f2s=f2s)

    with pytest.raises(ValueError, match="`form` is not recognised."):
        results.get_results(form="not_a_form")

    with pytest.raises(TypeError, match="`copy` must be a bool."):
        results.get_results(copy="True")


def test_results_cfc_runs() -> None:
    """Test `ResultsCFC` runs with correct inputs."""
    n_cons = 9
    n_f1 = 50
    n_f2 = 50
    data = _generate_data(n_cons, n_f1, n_f2)
    f1s = np.arange(n_f1)
    f2s = np.arange(n_f2)
    name = "test"
    n_unique_chans = 3
    indices = (
        tuple(np.repeat(np.arange(n_unique_chans), n_unique_chans).tolist()),
        tuple(np.tile(np.arange(n_unique_chans), n_unique_chans).tolist()),
    )

    results = ResultsCFC(data=data, indices=indices, f1s=f1s, f2s=f2s, name=name)

    assert repr(results) == (
        f"'<Result: {name} | [{n_cons} nodes, {n_f1} f1s, {n_f2} f2s]>'"
    )

    results_array = results.get_results(form="raveled")
    assert isinstance(results_array, np.ndarray)
    assert results_array.shape == (n_cons, n_f1, n_f2)

    results_array, array_indices = results.get_results(form="compact")
    assert isinstance(results_array, np.ndarray)
    assert results_array.shape == (n_unique_chans, n_unique_chans, n_f1, n_f2)
    assert array_indices == indices

    # Try also with unordered indices that aren't just [0, ..., n_chans]
    subset_n = 4
    subset = np.random.RandomState().choice(range(n_cons), subset_n, replace=False)
    indices_subset = (
        tuple(indices[0][idx] for idx in subset),
        tuple(indices[1][idx] for idx in subset),
    )
    results = ResultsCFC(
        data=data[subset], indices=indices_subset, f1s=f1s, f2s=f2s, name=name
    )
    results_array, array_indices = results.get_results(form="compact")
    assert np.max(array_indices) == len(np.unique(indices_subset)) - 1
    # Check that the results array is NaN for missing nodes
    array_subset_mask = np.zeros(results_array.shape[:2], dtype=bool)
    array_subset_mask[array_indices[0], array_indices[1]] = True
    assert not np.isnan(results_array[array_subset_mask]).all()
    assert np.isnan(results_array[~array_subset_mask]).all()


def test_results_tde_error_catch() -> None:
    """Test `ResultsTDE` catches errors."""
    n_cons = 9
    n_fbands = 2
    n_times = 50
    data = _generate_data(n_cons, n_fbands, n_times)
    times = np.arange(n_times)
    freq_bands = ((5, 10), (20, 30))
    n_unique_chans = 3
    indices = (
        tuple(np.repeat(np.arange(n_unique_chans), n_unique_chans).tolist()),
        tuple(np.tile(np.arange(n_unique_chans), n_unique_chans).tolist()),
    )

    with pytest.raises(TypeError, match="`data` must be a NumPy array."):
        ResultsTDE(
            data=data.tolist(), indices=indices, freq_bands=freq_bands, times=times
        )
    with pytest.raises(ValueError, match="`data` must be a 3D array."):
        ResultsTDE(
            data=data[..., 0], indices=indices, freq_bands=freq_bands, times=times
        )

    with pytest.raises(TypeError, match="`indices` must be a tuple."):
        ResultsTDE(data=data, indices=list(indices), freq_bands=freq_bands, times=times)
    with pytest.raises(ValueError, match="`indices` must have length of 2."):
        ResultsTDE(
            data=data,
            indices=(indices[0], indices[0], indices[1]),
            freq_bands=freq_bands,
            times=times,
        )
    with pytest.raises(TypeError, match="Entries of `indices` must be tuples."):
        ResultsTDE(data=data, indices=(0, 1), freq_bands=freq_bands, times=times)
    with pytest.raises(
        TypeError, match="Entries for seeds and targets in `indices` must be ints."
    ):
        ResultsTDE(
            data=data, indices=((0.5,), (1.5,)), freq_bands=freq_bands, times=times
        )
    with pytest.raises(
        ValueError, match="Entries of `indices` must have equal length."
    ):
        ResultsTDE(
            data=data,
            indices=(indices[0], tuple(np.concatenate((indices[1], [1])).tolist())),
            freq_bands=freq_bands,
            times=times,
        )
    with pytest.raises(
        ValueError,
        match=("Entries for seeds and targets in `indices` must be >= 0."),
    ):
        ResultsTDE(data=data, indices=((0,), (-1,)), freq_bands=freq_bands, times=times)

    with pytest.raises(TypeError, match="`freq_bands` must be a tuple."):
        ResultsTDE(data=data, indices=indices, freq_bands=list(freq_bands), times=times)
    with pytest.raises(
        ValueError,
        match=(
            "`freq_bands` must the same length as the number of frequency bands in the "
            "results."
        ),
    ):
        ResultsTDE(data=data, indices=indices, freq_bands=((5, 10),), times=times)
    with pytest.raises(TypeError, match="Each entry of `freq_bands` must be a tuple."):
        ResultsTDE(
            data=data,
            indices=indices,
            freq_bands=tuple(list(fband) for fband in freq_bands),
            times=times,
        )
    with pytest.raises(
        ValueError, match="Each entry of `freq_bands` must have length of 2."
    ):
        ResultsTDE(
            data=data,
            indices=indices,
            freq_bands=tuple((*fband, 0) for fband in freq_bands),
            times=times,
        )

    with pytest.raises(TypeError, match="`times` must be a NumPy array."):
        ResultsTDE(
            data=data, indices=indices, freq_bands=freq_bands, times=times.tolist()
        )
    with pytest.raises(ValueError, match="`times` must be a 1D array."):
        ResultsTDE(
            data=data,
            indices=indices,
            freq_bands=freq_bands,
            times=times[:, np.newaxis],
        )

    with pytest.raises(
        ValueError, match=r"`data` must have shape \[nodes, frequency bands, times\]."
    ):
        ResultsTDE(
            data=data[1:, :], indices=indices, freq_bands=freq_bands, times=times
        )
    with pytest.raises(
        ValueError, match=r"`data` must have shape \[nodes, frequency bands, times\]."
    ):
        ResultsTDE(data=data, indices=indices, freq_bands=freq_bands, times=times[1:])

    with pytest.raises(TypeError, match="`name` must be a string."):
        ResultsTDE(
            data=data, indices=indices, times=times, freq_bands=freq_bands, name=1
        )

    results = ResultsTDE(data=data, indices=indices, times=times)

    with pytest.raises(ValueError, match="`form` is not recognised."):
        results.get_results(form="not_a_form")

    with pytest.raises(TypeError, match="`copy` must be a bool."):
        results.get_results(copy="True")


@pytest.mark.parametrize("freq_bands", [None, ((5, 15),), ((5, 15), (10, 20))])
def test_results_tde_runs(freq_bands: tuple) -> None:
    """Test `ResultsTDE` runs with correct inputs."""
    n_cons = 9
    n_fbands = len(freq_bands) if freq_bands is not None else 1
    n_times = 50
    data = _generate_data(n_cons, n_fbands, n_times)
    times = np.arange(n_times)
    name = "test"
    n_unique_chans = 3
    indices = (
        tuple(np.repeat(np.arange(n_unique_chans), n_unique_chans).tolist()),
        tuple(np.tile(np.arange(n_unique_chans), n_unique_chans).tolist()),
    )

    results = ResultsTDE(
        data=data, indices=indices, times=times, name=name, freq_bands=freq_bands
    )

    if freq_bands is None:
        assert repr(results) == (
            f"<Result: {name} | [{n_cons} nodes, {n_fbands} frequency bands, "
            f"{n_times} times]>"
        )
    else:
        assert repr(results) == (
            f"<Result: {name} | {np.min(freq_bands):.2f} - {np.max(freq_bands):.2f} Hz "
            f"| [{n_cons} nodes, {n_fbands} frequency bands, {n_times} times]>"
        )

    results_array = results.get_results(form="raveled")
    assert isinstance(results_array, np.ndarray)
    assert results_array.shape == (n_cons, n_fbands, n_times)

    results_array, array_indices = results.get_results(form="compact")
    assert isinstance(results_array, np.ndarray)
    assert results_array.shape == (n_unique_chans, n_unique_chans, n_fbands, n_times)
    assert array_indices == indices

    # Try also with unordered indices that aren't just [0, ..., n_chans]
    subset_n = 4
    subset = np.random.RandomState().choice(range(n_cons), subset_n, replace=False)
    indices_subset = (
        tuple(indices[0][idx] for idx in subset),
        tuple(indices[1][idx] for idx in subset),
    )
    results = ResultsTDE(
        data=data[subset],
        indices=indices_subset,
        times=times,
        name=name,
        freq_bands=freq_bands,
    )

    results_array, array_indices = results.get_results(form="compact")

    assert np.max(array_indices) == len(np.unique(indices_subset)) - 1
    # Check that the results array is NaN for missing nodes
    array_subset_mask = np.zeros(results_array.shape[:2], dtype=bool)
    array_subset_mask[array_indices[0], array_indices[1]] = True
    assert not np.isnan(results_array[array_subset_mask]).all()
    assert np.isnan(results_array[~array_subset_mask]).all()


def test_results_waveshape_error_catch() -> None:
    """Test `ResultsWaveShape` catches errors."""
    n_chans = 3
    n_f1 = 50
    n_f2 = 50
    data = _generate_data(n_chans, n_f1, n_f2)
    f1s = np.arange(n_f1)
    f2s = np.arange(n_f2)
    indices = tuple(range(n_chans))

    with pytest.raises(TypeError, match="`data` must be a NumPy array."):
        ResultsWaveShape(data=data.tolist(), indices=indices, f1s=f1s, f2s=f2s)
    with pytest.raises(ValueError, match="`data` must be a 3D array."):
        ResultsWaveShape(data=data[..., 0], indices=indices, f1s=f1s, f2s=f2s)

    with pytest.raises(TypeError, match="`indices` must be a tuple."):
        ResultsWaveShape(data=data, indices=list(indices), f1s=f1s, f2s=f2s)
    with pytest.raises(TypeError, match="Entries of `indices` must be ints."):
        ResultsWaveShape(data=data, indices=(0.5,), f1s=f1s, f2s=f2s)
    with pytest.raises(
        ValueError,
        match=("Entries of `indices` must be >= 0."),
    ):
        ResultsWaveShape(data=data, indices=(-1,), f1s=f1s, f2s=f2s)

    with pytest.raises(TypeError, match="`f1s` and `f2s` must be NumPy arrays."):
        ResultsWaveShape(data=data, indices=indices, f1s=f1s.tolist(), f2s=f2s)
    with pytest.raises(TypeError, match="`f1s` and `f2s` must be NumPy arrays."):
        ResultsWaveShape(data=data, indices=indices, f1s=f1s, f2s=f2s.tolist())
    with pytest.raises(ValueError, match="`f1s` and `f2s` must be 1D arrays."):
        ResultsWaveShape(data=data, indices=indices, f1s=np.vstack((f1s, f1s)), f2s=f2s)
    with pytest.raises(ValueError, match="`f1s` and `f2s` must be 1D arrays."):
        ResultsWaveShape(data=data, indices=indices, f1s=f1s, f2s=np.vstack((f2s, f2s)))

    with pytest.raises(
        ValueError, match=r"`data` must have shape \[nodes, f1s, f2s\]."
    ):
        ResultsWaveShape(data=data[1:, :, :], indices=indices, f1s=f1s, f2s=f2s)
    with pytest.raises(
        ValueError, match=r"`data` must have shape \[nodes, f1s, f2s\]."
    ):
        ResultsWaveShape(data=data, indices=indices, f1s=f1s[1:], f2s=f2s)
    with pytest.raises(
        ValueError, match=r"`data` must have shape \[nodes, f1s, f2s\]."
    ):
        ResultsWaveShape(data=data, indices=indices, f1s=f1s, f2s=f2s[1:])

    with pytest.raises(TypeError, match="`name` must be a string."):
        ResultsWaveShape(data=data, indices=indices, f1s=f1s, f2s=f2s, name=1)

    results = ResultsWaveShape(data=data, indices=indices, f1s=f1s, f2s=f2s)

    with pytest.raises(TypeError, match="`copy` must be a bool."):
        results.get_results(copy="True")


def test_results_waveshape_runs() -> None:
    """Test `ResultsWaveShape` runs with correct inputs."""
    n_chans = 3
    n_f1 = 50
    n_f2 = 50
    data = _generate_data(n_chans, n_f1, n_f2)
    f1s = np.arange(n_f1)
    f2s = np.arange(n_f2)
    name = "test"
    indices = tuple(range(n_chans))

    results = ResultsWaveShape(data=data, indices=indices, f1s=f1s, f2s=f2s, name=name)

    assert repr(results) == (
        f"'<Result: {name} | [{n_chans} nodes, {n_f1} f1s, {n_f2} f2s]>'"
    )

    results_array = results.get_results(copy=True)
    results_array = results.get_results(copy=False)
    assert isinstance(results_array, np.ndarray)
    assert results_array.shape == (n_chans, n_f1, n_f2)


def test_results_general_error_catch() -> None:
    """Test `ResultsGeneral` catches errors."""
    n_chans = 27
    n_f1 = 50
    n_f2 = 50
    data = _generate_data(n_chans, n_f1, n_f2)
    f1s = np.arange(n_f1)
    f2s = np.arange(n_f2)
    n_unique_chans = 3
    indices = tuple(
        [
            tuple(np.tile(range(n_unique_chans), n_unique_chans**2).tolist()),
            tuple(
                np.repeat(
                    np.tile(range(n_unique_chans), n_unique_chans), n_unique_chans
                ).tolist()
            ),
            tuple(np.repeat(range(n_unique_chans), n_unique_chans**2).tolist()),
        ]
    )

    with pytest.raises(TypeError, match="`data` must be a NumPy array."):
        ResultsGeneral(data=data.tolist(), indices=indices, f1s=f1s, f2s=f2s)
    with pytest.raises(ValueError, match="`data` must be a 3D array."):
        ResultsGeneral(data=data[..., 0], indices=indices, f1s=f1s, f2s=f2s)

    with pytest.raises(TypeError, match="`indices` must be a tuple."):
        ResultsGeneral(data=data, indices=list(indices), f1s=f1s, f2s=f2s)
    with pytest.raises(ValueError, match="`indices` must have length of 3."):
        ResultsGeneral(data=data, indices=(indices[0], indices[1]), f1s=f1s, f2s=f2s)
    with pytest.raises(TypeError, match="Entries of `indices` must be tuples."):
        ResultsGeneral(data=data, indices=(0, 1, 2), f1s=f1s, f2s=f2s)
    with pytest.raises(
        TypeError, match="Entries for groups in `indices` must be ints."
    ):
        ResultsGeneral(data=data, indices=((0.5,), (1.5,), (2.5,)), f1s=f1s, f2s=f2s)
    with pytest.raises(
        ValueError,
        match=("Entries for groups in `indices` must be >= 0."),
    ):
        ResultsGeneral(data=data, indices=((-1,), (0,), (1,)), f1s=f1s, f2s=f2s)
    with pytest.raises(
        ValueError, match=("Entries of `indices` must have equal length.")
    ):
        ResultsGeneral(data=data, indices=((0,), (1,), (2, 3)), f1s=f1s, f2s=f2s)

    with pytest.raises(TypeError, match="`f1s` and `f2s` must be NumPy arrays."):
        ResultsGeneral(data=data, indices=indices, f1s=f1s.tolist(), f2s=f2s)
    with pytest.raises(TypeError, match="`f1s` and `f2s` must be NumPy arrays."):
        ResultsGeneral(data=data, indices=indices, f1s=f1s, f2s=f2s.tolist())
    with pytest.raises(ValueError, match="`f1s` and `f2s` must be 1D arrays."):
        ResultsGeneral(data=data, indices=indices, f1s=np.vstack((f1s, f1s)), f2s=f2s)
    with pytest.raises(ValueError, match="`f1s` and `f2s` must be 1D arrays."):
        ResultsGeneral(data=data, indices=indices, f1s=f1s, f2s=np.vstack((f2s, f2s)))

    with pytest.raises(
        ValueError, match=r"`data` must have shape \[nodes, f1s, f2s\]."
    ):
        ResultsGeneral(data=data[1:, :, :], indices=indices, f1s=f1s, f2s=f2s)
    with pytest.raises(
        ValueError, match=r"`data` must have shape \[nodes, f1s, f2s\]."
    ):
        ResultsGeneral(data=data, indices=indices, f1s=f1s[1:], f2s=f2s)
    with pytest.raises(
        ValueError, match=r"`data` must have shape \[nodes, f1s, f2s\]."
    ):
        ResultsGeneral(data=data, indices=indices, f1s=f1s, f2s=f2s[1:])

    with pytest.raises(TypeError, match="`name` must be a string."):
        ResultsGeneral(data=data, indices=indices, f1s=f1s, f2s=f2s, name=1)

    results = ResultsGeneral(data=data, indices=indices, f1s=f1s, f2s=f2s)

    with pytest.raises(ValueError, match="`form` is not recognised."):
        results.get_results(form="not_a_form")

    with pytest.raises(TypeError, match="`copy` must be a bool."):
        results.get_results(copy="True")


def test_results_general_runs() -> None:
    """Test `ResultsGeneral` runs with correct inputs."""
    n_cons = 27
    n_f1 = 50
    n_f2 = 50
    data = _generate_data(n_cons, n_f1, n_f2)
    f1s = np.arange(n_f1)
    f2s = np.arange(n_f2)
    name = "test"
    n_unique_chans = 3
    indices = tuple(
        [
            tuple(np.tile(range(n_unique_chans), n_unique_chans**2).tolist()),
            tuple(
                np.repeat(
                    np.tile(range(n_unique_chans), n_unique_chans), n_unique_chans
                ).tolist()
            ),
            tuple(np.repeat(range(n_unique_chans), n_unique_chans**2).tolist()),
        ]
    )

    results = ResultsGeneral(data=data, indices=indices, f1s=f1s, f2s=f2s, name=name)

    assert repr(results) == (
        f"'<Result: {name} | [{n_cons} nodes, {n_f1} f1s, {n_f2} f2s]>'"
    )

    results_array = results.get_results(form="raveled")
    assert isinstance(results_array, np.ndarray)
    assert results_array.shape == (n_cons, n_f1, n_f2)

    results_array, array_indices = results.get_results(form="compact")
    assert isinstance(results_array, np.ndarray)
    assert results_array.shape == (
        n_unique_chans,
        n_unique_chans,
        n_unique_chans,
        n_f1,
        n_f2,
    )
    assert array_indices == indices

    # Try also with unordered indices that aren't just [0, ..., n_chans]
    subset_n = 8
    subset = np.random.RandomState().choice(range(n_cons), subset_n, replace=False)
    indices_subset = (
        tuple(indices[0][idx] for idx in subset),
        tuple(indices[1][idx] for idx in subset),
        tuple(indices[2][idx] for idx in subset),
    )
    results = ResultsGeneral(
        data=data[subset], indices=indices_subset, f1s=f1s, f2s=f2s, name=name
    )
    results_array, array_indices = results.get_results(form="compact")
    assert np.max(array_indices) == len(np.unique(indices_subset)) - 1
    # Check that the results array is NaN for missing nodes
    array_subset_mask = np.zeros(results_array.shape[:3], dtype=bool)
    array_subset_mask[array_indices[0], array_indices[1], array_indices[2]] = True
    assert not np.isnan(results_array[array_subset_mask]).all()
    assert np.isnan(results_array[~array_subset_mask]).all()
