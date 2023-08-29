"""Tests for toolbox utilities."""

import warnings

import pytest
import numpy as np

from pybispectra.utils import (
    ResultsCFC,
    ResultsTDE,
    ResultsWaveShape,
    compute_fft,
)
from pybispectra.utils._utils import _fast_find_first, _generate_data


def test_results() -> None:
    """Test `ResultsCFC`."""
    n_cons = 9
    n_f1 = 50
    n_f2 = 50
    data = _generate_data(n_cons, n_f1, n_f2)

    f1s = np.arange(n_f1)
    f2s = np.arange(n_f2)

    n_unique_chans = 3
    indices = (
        np.repeat(np.arange(n_unique_chans), n_unique_chans),
        np.tile(np.arange(n_unique_chans), n_unique_chans),
    )

    # check if it runs with correct inputs
    results = ResultsCFC(
        data=data,
        indices=indices,
        f1s=f1s,
        f2s=f2s,
        name="test",
    )

    # check repr
    assert repr(results) == (
        f"'<Result: test | [{n_cons} nodes x {n_f1} f1s x {n_f2} f2s]>'"
    )

    # check if it catches incorrect inputs
    with pytest.raises(TypeError, match="`data` must be a NumPy array."):
        ResultsCFC(
            data=data.tolist(),
            indices=indices,
            f1s=f1s,
            f2s=f2s,
            name="test",
        )
    with pytest.raises(ValueError, match="`data` must be a 3D array."):
        ResultsCFC(
            data=data[:, :, 0],
            indices=indices,
            f1s=f1s,
            f2s=f2s,
            name="test",
        )

    with pytest.raises(TypeError, match="`indices` must be a tuple."):
        ResultsCFC(
            data=data,
            indices=list(indices),
            f1s=f1s,
            f2s=f2s,
            name="test",
        )
    with pytest.raises(ValueError, match="`indices` must have a length of 2."):
        ResultsCFC(
            data=data,
            indices=(indices[0], indices[0], indices[1]),
            f1s=f1s,
            f2s=f2s,
            name="test",
        )
    with pytest.raises(
        TypeError, match="Entries of `indices` must be NumPy arrays."
    ):
        ResultsCFC(
            data=data,
            indices=(indices[0][0], indices[1][0]),
            f1s=f1s,
            f2s=f2s,
            name="test",
        )
    with pytest.raises(
        ValueError, match="Entries of `indices` must be 1D arrays."
    ):
        ResultsCFC(
            data=data,
            indices=(
                np.vstack((indices[0], indices[0])),
                np.vstack((indices[1], indices[1])),
            ),
            f1s=f1s,
            f2s=f2s,
            name="test",
        )
    with pytest.raises(
        ValueError, match="Entries of `indices` must have the same length."
    ):
        ResultsCFC(
            data=data,
            indices=(indices[0], np.concatenate((indices[1], [1]))),
            f1s=f1s,
            f2s=f2s,
            name="test",
        )

    with pytest.raises(
        TypeError, match="`f1s` and `f2s` must be NumPy arrays."
    ):
        ResultsCFC(
            data=data,
            indices=indices,
            f1s=f1s.tolist(),
            f2s=f2s,
            name="test",
        )
    with pytest.raises(
        TypeError, match="`f1s` and `f2s` must be NumPy arrays."
    ):
        ResultsCFC(
            data=data,
            indices=indices,
            f1s=f1s,
            f2s=f2s.tolist(),
            name="test",
        )
    with pytest.raises(ValueError, match="`f1s` and `f2s` must be 1D arrays."):
        ResultsCFC(
            data=data,
            indices=indices,
            f1s=np.vstack((f1s, f1s)),
            f2s=f2s,
            name="test",
        )
    with pytest.raises(ValueError, match="`f1s` and `f2s` must be 1D arrays."):
        ResultsCFC(
            data=data,
            indices=indices,
            f1s=f1s,
            f2s=np.vstack((f2s, f2s)),
            name="test",
        )

    with pytest.raises(
        ValueError,
        match=r"`data` must have shape \[nodes x f1s x f2s\].",
    ):
        ResultsCFC(
            data=data[1:, :, :],
            indices=indices,
            f1s=f1s,
            f2s=f2s,
            name="test",
        )

    with pytest.raises(TypeError, match="`name` must be a string."):
        ResultsCFC(
            data=data,
            indices=indices,
            f1s=f1s,
            f2s=f2s,
            name=1,
        )

    # check if `get_results` works with correct inputs
    output = results.get_results(form="raveled")
    assert output.shape == (n_cons, n_f1, n_f2)
    output = results.get_results(form="compact")
    assert output[0].shape == (n_unique_chans, n_unique_chans, n_f1, n_f2)
    assert (idcs == np.arange(n_unique_chans) for idcs in output[1])

    # check if `get_results` catches incorrect inputs
    with pytest.raises(ValueError, match="`form` is not recognised."):
        results.get_results(form="notaform")

    # check if `plot` works with correct inputs
    figures, axes = results.plot(show=False)
    assert len(figures) == n_cons
    assert len(axes) == n_cons
    figures, axes = results.plot(n_rows=3, n_cols=3, show=True)
    assert len(figures) == 1
    assert len(axes) == 1

    # check if `plot` works with incorrect inputs
    with pytest.raises(TypeError, match="`nodes` must be a list of integers."):
        results.plot(nodes=9, show=False)
    with pytest.raises(TypeError, match="`nodes` must be a list of integers."):
        results.plot(nodes=[float(i) for i in range(n_cons)], show=False)
    with pytest.raises(
        ValueError, match="The requested connection is not present in the"
    ):
        results.plot(nodes=[-1], show=False)

    with pytest.raises(
        TypeError, match="`f1s` and `f2s` must be NumPy arrays."
    ):
        results.plot(f1s=0, show=False)
    with pytest.raises(
        TypeError, match="`f1s` and `f2s` must be NumPy arrays."
    ):
        results.plot(f2s=0, show=False)

    with pytest.raises(ValueError, match="`f1s` and `f2s` must be 1D arrays."):
        results.plot(f1s=np.random.rand(2, 2), show=False)
    with pytest.raises(ValueError, match="`f1s` and `f2s` must be 1D arrays."):
        results.plot(f2s=np.random.rand(2, 2), show=False)

    with pytest.raises(
        ValueError, match="Entries of `f1s` and `f2s` must be present in the"
    ):
        results.plot(f1s=f1s + 1, show=False)
    with pytest.raises(
        ValueError, match="Entries of `f1s` and `f2s` must be present in the"
    ):
        results.plot(f2s=f2s + 1, show=False)

    with pytest.raises(
        TypeError, match="`n_rows` and `n_cols` must be integers."
    ):
        results.plot(n_rows=0.5, show=False)
    with pytest.raises(
        TypeError, match="`n_rows` and `n_cols` must be integers."
    ):
        results.plot(n_cols=0.5, show=False)

    with pytest.raises(
        ValueError, match="`n_rows` and `n_cols` must be >= 1."
    ):
        results.plot(n_rows=0, show=False)
    with pytest.raises(
        ValueError, match="`n_rows` and `n_cols` must be >= 1."
    ):
        results.plot(n_cols=0, show=False)

    with pytest.raises(
        TypeError, match="`major_tick_intervals` and `minor_tick_intervals`"
    ):
        results.plot(major_tick_intervals="5", show=False)

    with pytest.raises(
        TypeError, match="`major_tick_intervals` and `minor_tick_intervals`"
    ):
        results.plot(minor_tick_intervals="1", show=False)


def test_compute_fft() -> None:
    """Test `compute_fft`."""
    n_epochs = 5
    n_chans = 3
    n_times = 100
    data = _generate_data(n_epochs, n_chans, n_times)
    sampling_freq = 50

    # check it runs with correct inputs
    fft, freqs = compute_fft(
        data=data, sampling_freq=sampling_freq, n_jobs=1, verbose=False
    )
    assert isinstance(fft, np.ndarray), "`fft` should be a NumPy array."
    assert fft.ndim == 3, "`fft` should have 3 dimensions."
    assert fft.shape[:2] == (n_epochs, n_chans), (
        "The first 2 dimensions of `fft` should have shape [epochs x "
        "channels]."
    )
    assert isinstance(freqs, np.ndarray), "`freqs` should be a NumPy array."
    assert freqs.ndim == 1, "`freqs` should have 1 dimension."
    assert (
        fft.shape[2] == freqs.shape[0]
    ), "The 3rd dimension of `fft` should have the same length as `freqs`."
    assert (
        freqs[-1] == sampling_freq / 2
    ), "The maximum of `freqs` should be the Nyquist frequency."
    assert freqs[0] != 0, "The zero frequency should not be included."

    # check it catches incorrect inputs
    with pytest.raises(TypeError, match="`data` must be a NumPy array."):
        compute_fft(data=data.tolist(), sampling_freq=sampling_freq)
    with pytest.raises(ValueError, match="`data` must be a 3D array."):
        compute_fft(data=data[..., 0], sampling_freq=sampling_freq)

    with pytest.raises(TypeError, match="`n_jobs` must be an integer."):
        compute_fft(data=data, sampling_freq=sampling_freq, n_jobs=[])
    with pytest.raises(ValueError, match="`n_jobs` must be >= 1."):
        compute_fft(data=data, sampling_freq=sampling_freq, n_jobs=0)

    # check it works with parallelisation (already tested without)
    compute_fft(
        data=data, sampling_freq=sampling_freq, n_jobs=2, verbose=False
    )

    # check a warning is raised if `data` is complex-valued
    with warnings.catch_warnings():
        compute_fft(
            data=np.array(data, dtype=np.complex128) + 1j,
            sampling_freq=sampling_freq,
        )


def test_fast_find_first():
    """Test `_fast_find_first`."""
    # test that a present value is found
    index = _fast_find_first(vector=np.array([-1, 0, 1, 2, 3, 1]), value=1)
    assert index == 2, "The index of the value being found should be 2."

    # test that a missing value is not found
    with pytest.raises(
        ValueError, match="`value` is not present in `vector`."
    ):
        _fast_find_first(vector=np.array([-1, 0, 2, 2, 3, 4]), value=1)
