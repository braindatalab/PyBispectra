"""Tests for results classes (plotting tested separately)."""

import pytest
import numpy as np

from pybispectra.utils import ResultsCFC, ResultsTDE, ResultsWaveShape
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
        ResultsCFC(
            data=data.tolist(),
            indices=indices,
            f1s=f1s,
            f2s=f2s,
        )
    with pytest.raises(ValueError, match="`data` must be a 3D array."):
        ResultsCFC(
            data=data[..., 0],
            indices=indices,
            f1s=f1s,
            f2s=f2s,
        )

    with pytest.raises(TypeError, match="`indices` must be a tuple."):
        ResultsCFC(
            data=data,
            indices=list(indices),
            f1s=f1s,
            f2s=f2s,
        )
    with pytest.raises(ValueError, match="`indices` must have length of 2."):
        ResultsCFC(
            data=data,
            indices=(indices[0], indices[0], indices[1]),
            f1s=f1s,
            f2s=f2s,
        )
    with pytest.raises(
        TypeError, match="Entries of `indices` must be tuples."
    ):
        ResultsCFC(
            data=data,
            indices=(0, 1),
            f1s=f1s,
            f2s=f2s,
        )
    with pytest.raises(
        TypeError,
        match="Entries for seeds and targets in `indices` must be ints.",
    ):
        ResultsCFC(
            data=data,
            indices=((0.5,), (1.5,)),
            f1s=f1s,
            f2s=f2s,
        )
    with pytest.raises(
        ValueError, match="Entries of `indices` must have equal length."
    ):
        ResultsCFC(
            data=data,
            indices=(
                indices[0],
                tuple(np.concatenate((indices[1], [1])).tolist()),
            ),
            f1s=f1s,
            f2s=f2s,
        )
    with pytest.raises(
        ValueError,
        match=(
            "`indices` contains indices for nodes not present in the data."
        ),
    ):
        ResultsCFC(
            data=data,
            indices=((0,), (n_cons + 1,)),
            f1s=f1s,
            f2s=f2s,
        )

    with pytest.raises(
        TypeError, match="`f1s` and `f2s` must be NumPy arrays."
    ):
        ResultsCFC(
            data=data,
            indices=indices,
            f1s=f1s.tolist(),
            f2s=f2s,
        )
    with pytest.raises(
        TypeError, match="`f1s` and `f2s` must be NumPy arrays."
    ):
        ResultsCFC(
            data=data,
            indices=indices,
            f1s=f1s,
            f2s=f2s.tolist(),
        )
    with pytest.raises(ValueError, match="`f1s` and `f2s` must be 1D arrays."):
        ResultsCFC(
            data=data,
            indices=indices,
            f1s=np.vstack((f1s, f1s)),
            f2s=f2s,
        )
    with pytest.raises(ValueError, match="`f1s` and `f2s` must be 1D arrays."):
        ResultsCFC(
            data=data,
            indices=indices,
            f1s=f1s,
            f2s=np.vstack((f2s, f2s)),
        )

    with pytest.raises(
        ValueError,
        match=r"`data` must have shape \[nodes, f1s, f2s\].",
    ):
        ResultsCFC(
            data=data[1:, :, :],
            indices=indices,
            f1s=f1s,
            f2s=f2s,
        )
    with pytest.raises(
        ValueError,
        match=r"`data` must have shape \[nodes, f1s, f2s\].",
    ):
        ResultsCFC(
            data=data,
            indices=indices,
            f1s=f1s[1:],
            f2s=f2s,
        )
    with pytest.raises(
        ValueError,
        match=r"`data` must have shape \[nodes, f1s, f2s\].",
    ):
        ResultsCFC(
            data=data,
            indices=indices,
            f1s=f1s,
            f2s=f2s[1:],
        )

    with pytest.raises(TypeError, match="`name` must be a string."):
        ResultsCFC(
            data=data,
            indices=indices,
            f1s=f1s,
            f2s=f2s,
            name=1,
        )

    results = ResultsCFC(data=data, indices=indices, f1s=f1s, f2s=f2s)

    with pytest.raises(ValueError, match="`form` is not recognised."):
        results.get_results(form="not_a_form")


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

    results = ResultsCFC(
        data=data, indices=indices, f1s=f1s, f2s=f2s, name=name
    )

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
        n_f1,
        n_f2,
    )
    assert array_indices == (
        tuple(range(n_unique_chans)),
        tuple(range(n_unique_chans)),
    )


def test_results_tde_error_catch() -> None:
    """Test `ResultsTDE` catches errors."""
    n_cons = 9
    n_times = 50
    data = _generate_data(n_cons, n_times, 1)[..., 0]
    times = np.arange(n_times)
    n_unique_chans = 3
    indices = (
        tuple(np.repeat(np.arange(n_unique_chans), n_unique_chans).tolist()),
        tuple(np.tile(np.arange(n_unique_chans), n_unique_chans).tolist()),
    )

    with pytest.raises(TypeError, match="`data` must be a NumPy array."):
        ResultsTDE(
            data=data.tolist(),
            indices=indices,
            times=times,
        )
    with pytest.raises(ValueError, match="`data` must be a 2D array."):
        ResultsTDE(
            data=data[..., 0],
            indices=indices,
            times=times,
        )

    with pytest.raises(TypeError, match="`indices` must be a tuple."):
        ResultsTDE(
            data=data,
            indices=list(indices),
            times=times,
        )
    with pytest.raises(ValueError, match="`indices` must have length of 2."):
        ResultsTDE(
            data=data,
            indices=(indices[0], indices[0], indices[1]),
            times=times,
        )
    with pytest.raises(
        TypeError, match="Entries of `indices` must be tuples."
    ):
        ResultsTDE(
            data=data,
            indices=(0, 1),
            times=times,
        )
    with pytest.raises(
        TypeError,
        match="Entries for seeds and targets in `indices` must be ints.",
    ):
        ResultsTDE(
            data=data,
            indices=((0.5,), (1.5,)),
            times=times,
        )
    with pytest.raises(
        ValueError, match="Entries of `indices` must have equal length."
    ):
        ResultsTDE(
            data=data,
            indices=(
                indices[0],
                tuple(np.concatenate((indices[1], [1])).tolist()),
            ),
            times=times,
        )
    with pytest.raises(
        ValueError,
        match=(
            "`indices` contains indices for nodes not present in the data."
        ),
    ):
        ResultsTDE(
            data=data,
            indices=((0,), (n_cons + 1,)),
            times=times,
        )

    with pytest.raises(TypeError, match="`times` must be a NumPy array."):
        ResultsTDE(
            data=data,
            indices=indices,
            times=times.tolist(),
        )
    with pytest.raises(ValueError, match="`times` must be a 1D array."):
        ResultsTDE(
            data=data,
            indices=indices,
            times=times[:, np.newaxis],
        )

    with pytest.raises(
        ValueError,
        match=r"`data` must have shape \[nodes, times\].",
    ):
        ResultsTDE(
            data=data[1:, :],
            indices=indices,
            times=times,
        )
    with pytest.raises(
        ValueError,
        match=r"`data` must have shape \[nodes, times\].",
    ):
        ResultsTDE(
            data=data,
            indices=indices,
            times=times[1:],
        )

    with pytest.raises(TypeError, match="`freq_band` must be a tuple."):
        ResultsTDE(data=data, indices=indices, times=times, freq_band=5)
    with pytest.raises(ValueError, match="`freq_band` must have length of 2."):
        ResultsTDE(
            data=data, indices=indices, times=times, freq_band=(5, 10, 15)
        )

    with pytest.raises(TypeError, match="`name` must be a string."):
        ResultsTDE(
            data=data,
            indices=indices,
            times=times,
            name=1,
        )

    results = ResultsTDE(data=data, indices=indices, times=times)

    with pytest.raises(ValueError, match="`form` is not recognised."):
        results.get_results(form="not_a_form")


def test_results_tde_runs() -> None:
    """Test `ResultsTDE` runs with correct inputs."""
    n_cons = 9
    n_times = 50
    data = _generate_data(n_cons, n_times, 1)[..., 0]
    times = np.arange(n_times)
    name = "test"
    freq_band = (10, 20)
    n_unique_chans = 3
    indices = (
        tuple(np.repeat(np.arange(n_unique_chans), n_unique_chans).tolist()),
        tuple(np.tile(np.arange(n_unique_chans), n_unique_chans).tolist()),
    )

    results = ResultsTDE(
        data=data,
        indices=indices,
        times=times,
        name=name,
        freq_band=None,
    )

    assert repr(results) == (
        f"<Result: {name} | [{n_cons} nodes, {n_times} times]>"
    )

    results = ResultsTDE(
        data=data,
        indices=indices,
        times=times,
        name=name,
        freq_band=freq_band,
    )

    assert repr(results) == (
        f"<Result: {name} | {freq_band[0]} - {freq_band[1]} Hz | [{n_cons} "
        f"nodes, {n_times} times]>"
    )

    results_array = results.get_results(form="raveled")
    assert isinstance(results_array, np.ndarray)
    assert results_array.shape == (n_cons, n_times)

    results_array, array_indices = results.get_results(form="compact")
    assert isinstance(results_array, np.ndarray)
    assert results_array.shape == (n_unique_chans, n_unique_chans, n_times)
    assert array_indices == (
        tuple(range(n_unique_chans)),
        tuple(range(n_unique_chans)),
    )


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
        ResultsWaveShape(
            data=data.tolist(),
            indices=indices,
            f1s=f1s,
            f2s=f2s,
        )
    with pytest.raises(ValueError, match="`data` must be a 3D array."):
        ResultsWaveShape(
            data=data[..., 0],
            indices=indices,
            f1s=f1s,
            f2s=f2s,
        )

    with pytest.raises(TypeError, match="`indices` must be a tuple."):
        ResultsWaveShape(
            data=data,
            indices=list(indices),
            f1s=f1s,
            f2s=f2s,
        )
    with pytest.raises(TypeError, match="Entries of `indices` must be ints."):
        ResultsWaveShape(
            data=data,
            indices=(0.5,),
            f1s=f1s,
            f2s=f2s,
        )
    with pytest.raises(
        ValueError,
        match=(
            "`indices` contains indices for channels not present in the data."
        ),
    ):
        ResultsWaveShape(
            data=data,
            indices=(-1,),
            f1s=f1s,
            f2s=f2s,
        )
    with pytest.raises(
        ValueError,
        match=(
            "`indices` contains indices for channels not present in the data."
        ),
    ):
        ResultsWaveShape(
            data=data,
            indices=(n_chans + 1,),
            f1s=f1s,
            f2s=f2s,
        )

    with pytest.raises(
        TypeError, match="`f1s` and `f2s` must be NumPy arrays."
    ):
        ResultsWaveShape(
            data=data,
            indices=indices,
            f1s=f1s.tolist(),
            f2s=f2s,
        )
    with pytest.raises(
        TypeError, match="`f1s` and `f2s` must be NumPy arrays."
    ):
        ResultsWaveShape(
            data=data,
            indices=indices,
            f1s=f1s,
            f2s=f2s.tolist(),
        )
    with pytest.raises(ValueError, match="`f1s` and `f2s` must be 1D arrays."):
        ResultsWaveShape(
            data=data,
            indices=indices,
            f1s=np.vstack((f1s, f1s)),
            f2s=f2s,
        )
    with pytest.raises(ValueError, match="`f1s` and `f2s` must be 1D arrays."):
        ResultsWaveShape(
            data=data,
            indices=indices,
            f1s=f1s,
            f2s=np.vstack((f2s, f2s)),
        )

    with pytest.raises(
        ValueError,
        match=r"`data` must have shape \[nodes, f1s, f2s\].",
    ):
        ResultsWaveShape(
            data=data[1:, :, :],
            indices=indices,
            f1s=f1s,
            f2s=f2s,
        )
    with pytest.raises(
        ValueError,
        match=r"`data` must have shape \[nodes, f1s, f2s\].",
    ):
        ResultsWaveShape(
            data=data,
            indices=indices,
            f1s=f1s[1:],
            f2s=f2s,
        )
    with pytest.raises(
        ValueError,
        match=r"`data` must have shape \[nodes, f1s, f2s\].",
    ):
        ResultsWaveShape(
            data=data,
            indices=indices,
            f1s=f1s,
            f2s=f2s[1:],
        )

    with pytest.raises(TypeError, match="`name` must be a string."):
        ResultsWaveShape(
            data=data,
            indices=indices,
            f1s=f1s,
            f2s=f2s,
            name=1,
        )


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

    results = ResultsWaveShape(
        data=data, indices=indices, f1s=f1s, f2s=f2s, name=name
    )

    assert repr(results) == (
        f"'<Result: {name} | [{n_chans} nodes, {n_f1} f1s, {n_f2} f2s]>'"
    )

    results_array = results.get_results()
    assert isinstance(results_array, np.ndarray)
    assert results_array.shape == (n_chans, n_f1, n_f2)
