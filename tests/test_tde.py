"""Tests for TDE tools."""

import pytest
import numpy as np

from pybispectra.data import get_example_data_paths
from pybispectra.tde import TDE
from pybispectra.utils import ResultsTDE, compute_fft
from pybispectra.utils._utils import _generate_data


def test_error_catch() -> None:
    """Check that the TDE class catches errors."""
    n_chans = 3
    n_epochs = 5
    n_times = 100
    sampling_freq = 50
    data = _generate_data(n_epochs, n_chans, n_times)
    indices = ((0, 1, 2), (0, 1, 2))

    coeffs, freqs = compute_fft(
        data=data,
        sampling_freq=sampling_freq,
        n_points=2 * n_times + 1,
        window="hamming",
    )

    # initialisation
    with pytest.raises(TypeError, match="`data` must be a NumPy array."):
        TDE(coeffs.tolist(), freqs, sampling_freq)
    with pytest.raises(ValueError, match="`data` must be a 3D array."):
        TDE(np.random.randn(2, 2), freqs, sampling_freq)

    with pytest.raises(TypeError, match="`freqs` must be a NumPy array."):
        TDE(coeffs, freqs.tolist(), sampling_freq)
    with pytest.raises(ValueError, match="`freqs` must be a 1D array."):
        TDE(coeffs, np.random.randn(2, 2), sampling_freq)
    with pytest.raises(
        ValueError, match="The first entry of `freqs` must be 0."
    ):
        TDE(coeffs[..., 1:], freqs[1:], sampling_freq)

    with pytest.raises(
        ValueError,
        match=(
            "`data` and `freqs` must contain the same number of frequencies."
        ),
    ):
        TDE(coeffs, freqs[:-1], sampling_freq)

    with pytest.raises(
        ValueError,
        match="At least one entry of `freqs` is > the Nyquist frequency.",
    ):
        bad_freqs = freqs.copy()
        bad_freqs[np.argmax(bad_freqs)] = sampling_freq / 2 + 1
        TDE(coeffs, bad_freqs, sampling_freq)
    max_freq_i = np.argwhere(freqs == np.max(freqs))[0][0]
    with pytest.raises(
        ValueError, match="Entries of `freqs` must be in ascending order."
    ):
        TDE(
            coeffs,
            np.hstack(
                (freqs[: max_freq_i + 1][::-1], freqs[max_freq_i + 1 :])
            ),
            sampling_freq,
        )

    with pytest.raises(
        TypeError, match="`sampling_freq` must be an int or a float."
    ):
        TDE(coeffs, freqs, None)

    with pytest.raises(TypeError, match="`verbose` must be a bool."):
        TDE(coeffs, freqs, sampling_freq, "verbose")

    # compute
    tde = TDE(coeffs, freqs, sampling_freq)

    with pytest.raises(
        TypeError, match="`antisym` must be a bool or tuple of bools."
    ):
        tde.compute(antisym="true")
    with pytest.raises(TypeError, match="Entries of `antisym` must be bools."):
        tde.compute(antisym=("true",))

    with pytest.raises(
        TypeError, match="`method` must be an int or tuple of ints."
    ):
        tde.compute(method=0.5)
    with pytest.raises(
        ValueError, match="The value of `method` is not recognised."
    ):
        tde.compute(method=0)
    with pytest.raises(
        ValueError, match="The value of `method` is not recognised."
    ):
        tde.compute(method=(1, 5))

    with pytest.raises(TypeError, match="`indices` must be a tuple."):
        tde.compute(indices=list(indices))
    with pytest.raises(ValueError, match="`indices` must have length of 2."):
        tde.compute(indices=(0, 1, 2))
    with pytest.raises(
        TypeError, match="Entries of `indices` must be tuples."
    ):
        tde.compute(indices=(0, 1))
    with pytest.raises(
        TypeError,
        match="Entries for seeds and targets in `indices` must be ints.",
    ):
        tde.compute(indices=((0.0,), (1.0,)))
    with pytest.raises(
        ValueError,
        match=(
            "`indices` contains indices for channels not present in the data."
        ),
    ):
        tde.compute(indices=((0,), (99,)))
    with pytest.raises(
        ValueError, match="Entries of `indices` must have equal length."
    ):
        tde.compute(indices=((0,), (1, 2)))
    with pytest.raises(
        ValueError,
        match=(
            "Seeds and targets in `indices` must not be the same channel for "
            "any connection."
        ),
    ):
        tde.compute(indices=((0,), (0,)))

    with pytest.raises(
        TypeError, match="`fmin` must be an int, float, or tuple."
    ):
        tde.compute(fmin="0")
    with pytest.raises(
        TypeError, match="`fmax` must be an int, float, or tuple."
    ):
        tde.compute(fmax="10")
    with pytest.raises(
        ValueError, match="`fmin` and `fmax` must have the same length."
    ):
        tde.compute(fmin=0, fmax=(10, 20))
    with pytest.raises(ValueError, match="Entries of `fmin` must be >= 0."):
        tde.compute(fmin=-1)
    with pytest.raises(
        ValueError, match="Entries of `fmax` must be <= the Nyquist frequency."
    ):
        tde.compute(fmax=sampling_freq / 2 + 1)
    with pytest.raises(
        ValueError,
        match=(
            "At least one entry of `fmin` is > the corresponding entry of "
            "`fmax`."
        ),
    ):
        tde.compute(fmin=(5, 20), fmax=(10, 15))
    with pytest.raises(
        ValueError,
        match=(
            r"No frequencies are present in the data for the range \(0.1, "
            r"0.2\)."
        ),
    ):
        tde.compute(fmin=0.1, fmax=0.2)

    with pytest.raises(TypeError, match="`n_jobs` must be an integer."):
        tde.compute(n_jobs=0.5)
    with pytest.raises(ValueError, match="`n_jobs` must be >= 1 or -1."):
        tde.compute(n_jobs=0)


@pytest.mark.parametrize(
    "freq_bands", [(0, np.inf), (5, 10), ((5, 15), (10, 20))]
)
def test_tde_runs(freq_bands: tuple) -> None:
    """Test that TDE runs correctly."""
    n_chans = 3
    n_times = 100
    sampling_freq = 50
    data = _generate_data(5, n_chans, n_times)

    fft, freqs = compute_fft(
        data=data,
        sampling_freq=sampling_freq,
        n_points=2 * n_times + 1,
        window="hamming",
    )

    # check it runs with correct inputs
    tde = TDE(data=fft, freqs=freqs, sampling_freq=sampling_freq)
    tde.compute(
        antisym=(False, True),
        method=(1, 2, 3, 4),
        fmin=freq_bands[0],
        fmax=freq_bands[1],
    )

    # check the returned results have the correct shape
    assert (
        results.shape == (n_chans * (n_chans - 1) / 2, results.times)
        for results in tde.results
    )

    # check the returned results are of the correct type
    result_types = [
        "TDE | Method I",
        "TDE | Method II",
        "TDE | Method III",
        "TDE | Method IV",
        "TDE (antisymmetrised) | Method I",
        "TDE (antisymmetrised) | Method II",
        "TDE (antisymmetrised) | Method III",
        "TDE (antisymmetrised) | Method IV",
    ]
    assert (
        results.name == result_types[i]
        for i, results in enumerate(tde.results)
    )
    assert (isinstance(results, ResultsTDE) for results in tde.results)

    for antisym_arg, symmetrise_name in zip(
        [False, True], ["", "(antisymmetrised) "]
    ):
        for method_arg, method_name in zip(
            [1, 2, 3, 4], ["I", "II", "III", "IV"]
        ):
            tde.compute(antisym=antisym_arg, method=method_arg)
            assert isinstance(tde.results, ResultsTDE)
            assert (
                tde.results.name
                == f"TDE {symmetrise_name}| Method {method_name}"
            )

    # test it runs with parallelisation
    tde.compute(n_jobs=2)
    tde.compute(n_jobs=-1)

    # test copying works
    tde_copy = tde.copy()
    attrs = tde.__dict__.keys()
    for attr in attrs:
        if not attr.startswith("_"):
            assert np.all(getattr(tde, attr) == getattr(tde_copy, attr))
    assert tde is not tde_copy


def test_tde_results():
    """Test that TDE returns the correct results.

    Simulated data with a 250 ms delay between channels is used. In the case
    that noise is correlated between channels, spurious TDE will be detected at
    0 ms, which should be corrected for using antisymmetrisation to reveal the
    true delay of 250 ms.
    """
    tau = 250.0  # ms

    # test that TDE is detected at 250 ms with independent noise
    # load simulated data with independent noise
    data = np.load(get_example_data_paths("sim_data_tde_independent_noise"))
    sampling_freq = 200  # Hz
    n_times = data.shape[2]

    # compute FFT
    fft_coeffs, freqs = compute_fft(
        data=data,
        sampling_freq=sampling_freq,
        n_points=2 * n_times + 1,
        window="hamming",
    )

    # compute TDE (seed -> target; target -> seed)
    tde = TDE(data=fft_coeffs, freqs=freqs, sampling_freq=sampling_freq)
    tde.compute(indices=((0, 1), (1, 0)), method=(1, 2, 3, 4))

    # check that 250 ms delay is detected (negative for target -> seed)
    assert (results.tau[0] == tau for results in tde.results)
    assert (results.tau[1] == -tau for results in tde.results)

    # test that TDE is detected at 250 ms with correlated noise (with antisym.)
    # load simulated data with correlated noise
    data = np.load(get_example_data_paths("sim_data_tde_correlated_noise"))
    sampling_freq = 200  # Hz
    n_times = data.shape[2]

    # compute FFT
    fft_coeffs, freqs = compute_fft(
        data=data,
        sampling_freq=sampling_freq,
        n_points=2 * n_times + 1,
        window="hamming",
    )

    # compute TDE
    tde = TDE(data=fft_coeffs, freqs=freqs, sampling_freq=sampling_freq)
    tde.compute(indices=((0,), (1,)), method=(1, 2, 3, 4), antisym=False)
    # check that 0 ms delay is dominant without antisymmetrisation
    assert (results.tau[0] == 0 for results in tde.results)

    tde.compute(indices=((0,), (1,)), method=(1, 2, 3, 4), antisym=True)
    # check that 250 ms delay is recovered with antisymmetrisation
    assert (results.tau[0] == tau for results in tde.results)
