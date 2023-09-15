"""Tests for results classes (plotting tested separately)."""

import pytest
import numpy as np

from pybispectra.utils import ResultsCFC, ResultsTDE, ResultsWaveShape
from pybispectra.utils._utils import _generate_data


def test_plotting_cfc_error_catch() -> None:
    """Test plotting in `ResultsCFC` catches errors."""
    n_cons = 9
    n_f1 = 50
    n_f2 = 50
    data = _generate_data(n_cons, n_f1, n_f2)
    f1s = np.arange(n_f1)
    f2s = np.arange(n_f2)
    name = "test"
    n_unique_chans = 3
    indices = (
        np.repeat(np.arange(n_unique_chans), n_unique_chans).tolist(),
        np.tile(np.arange(n_unique_chans), n_unique_chans).tolist(),
    )

    results = ResultsCFC(
        data=data,
        indices=indices,
        f1s=f1s,
        f2s=f2s,
        name=name,
    )

    with pytest.raises(TypeError, match="`nodes` must be a list of integers."):
        results.plot(nodes=9, show=False)
    with pytest.raises(TypeError, match="`nodes` must be a list of integers."):
        results.plot(nodes=[float(i) for i in range(n_cons)], show=False)
    with pytest.raises(
        ValueError, match="The requested node is not present in the results."
    ):
        results.plot(nodes=[-1], show=False)

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
        ValueError,
        match="Entries of `f1s` and `f2s` must be present in the results.",
    ):
        results.plot(f1s=f1s + 1, show=False)
    with pytest.raises(
        ValueError,
        match="Entries of `f1s` and `f2s` must be present in the results",
    ):
        results.plot(f2s=f2s + 1, show=False)

    with pytest.raises(
        TypeError,
        match=(
            "`major_tick_intervals` and `minor_tick_intervals` should be ints "
            "or floats."
        ),
    ):
        results.plot(major_tick_intervals="5", show=False)
    with pytest.raises(
        TypeError,
        match=(
            "`major_tick_intervals` and `minor_tick_intervals` should be ints "
            "or floats."
        ),
    ):
        results.plot(minor_tick_intervals="1", show=False)
    with pytest.raises(
        ValueError,
        match=(
            r"`major_tick_intervals` and `minor_tick_intervals` should be \> "
            "0."
        ),
    ):
        results.plot(major_tick_intervals=0, show=False)
    with pytest.raises(
        ValueError,
        match=(
            r"`major_tick_intervals` and `minor_tick_intervals` should be \> "
            "0."
        ),
    ):
        results.plot(minor_tick_intervals=0, show=False)
    with pytest.raises(
        ValueError,
        match=r"`major_tick_intervals` should be \> `minor_tick_intervals`.",
    ):
        results.plot(
            major_tick_intervals=5, minor_tick_intervals=7, show=False
        )

    with pytest.raises(
        TypeError, match="`cbar_range` must be a list, tuple, or None."
    ):
        results.plot(cbar_range=np.array([0, 1]), show=False)
    with pytest.raises(
        ValueError,
        match=(
            "If `cbar_range` is a tuple, one entry must be provided for each "
            "node being plotted."
        ),
    ):
        results.plot(cbar_range=(None,), show=False)
    with pytest.raises(
        ValueError, match="Limits in `cbar_range` must have length of 2."
    ):
        results.plot(cbar_range=[0, 1, 2], show=False)
    with pytest.raises(
        ValueError, match="Limits in `cbar_range` must have length of 2."
    ):
        results.plot(
            cbar_range=tuple([0, 1, 2] for _ in range(n_cons)), show=False
        )


@pytest.mark.skip(reason="WIP")
def test_plotting_cfc() -> None:
    """Test plotting in `ResultsCFC` catches errors."""
    n_cons = 9
    n_f1 = 50
    n_f2 = 50
    data = _generate_data(n_cons, n_f1, n_f2)
    f1s = np.arange(n_f1)
    f2s = np.arange(n_f2)
    name = "test"
    n_unique_chans = 3
    indices = (
        np.repeat(np.arange(n_unique_chans), n_unique_chans).tolist(),
        np.tile(np.arange(n_unique_chans), n_unique_chans).tolist(),
    )

    results = ResultsCFC(
        data=data,
        indices=indices,
        f1s=f1s,
        f2s=f2s,
        name=name,
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
