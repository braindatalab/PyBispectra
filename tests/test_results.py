"""Tests for results classes."""

import pytest
import numpy as np

from pybispectra.utils import ResultsCFC, ResultsTDE, ResultsWaveShape
from pybispectra.utils._utils import _generate_data


@pytest.mark.skip(reason="test incomplete")
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
        np.repeat(np.arange(n_unique_chans), n_unique_chans).tolist(),
        np.tile(np.arange(n_unique_chans), n_unique_chans).tolist(),
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
        f"'<Result: test | [{n_cons} nodes, {n_f1} f1s, {n_f2} f2s]>'"
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
        match=r"`data` must have shape \[nodes, f1s, f2s\].",
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
