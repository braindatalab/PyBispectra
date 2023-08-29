"""Tests for TDE tools."""

import os

import pytest
import numpy as np

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
    indices = ([0, 1, 2], [0, 1, 2])

    coeffs, freqs = compute_fft(
        data=data,
        sampling_freq=sampling_freq,
        n_points=2 * n_times + 1,
        window="hamming",
        return_neg_freqs=True,
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
            "`freqs` must have an odd number of entries, consisting of the "
            "positive frequencies, the corresponding negative frequencies, "
            "and the zero frequency."
        ),
    ):
        TDE(coeffs[..., :-1], freqs[:-1], sampling_freq)
    with pytest.raises(
        ValueError,
        match=(
            "Each positive frequency in `freqs` must have a corresponding "
            "negative frequency."
        ),
    ):
        TDE(coeffs, np.hstack((freqs[:-1], -freqs[-1])), sampling_freq)

    with pytest.raises(
        ValueError,
        match=(
            "`data` and `freqs` must contain the same number of frequencies."
        ),
    ):
        TDE(coeffs, freqs[:-1], sampling_freq)

    max_freq_i = np.argwhere(freqs == np.max(freqs))[0][0]
    with pytest.raises(
        ValueError,
        match=(
            "Entries of `freqs` corresponding to positive frequencies must be "
            "in ascending order."
        ),
    ):
        TDE(
            coeffs,
            np.hstack(
                (freqs[: max_freq_i + 1][::-1], freqs[max_freq_i + 1 :])
            ),
            sampling_freq,
        )
    with pytest.raises(
        ValueError,
        match=(
            "Entries of `freqs` corresponding to negative frequencies must be "
            "in ascending order."
        ),
    ):
        TDE(
            coeffs,
            np.hstack(
                (freqs[: max_freq_i + 1], freqs[max_freq_i + 1 :][::-1])
            ),
            sampling_freq,
        )
    with pytest.raises(
        ValueError,
        match=(
            "Entries of `freqs` must have the form positive frequencies, then "
            "negative frequencies."
        ),
    ):
        TDE(coeffs, freqs[::-1], sampling_freq)

    with pytest.raises(TypeError, match="`verbose` must be a bool."):
        TDE(coeffs, freqs, sampling_freq, "verbose")

    # compute
    tde = TDE(coeffs, freqs, sampling_freq)

    with pytest.raises(
        TypeError, match="`symmetrise` must be a list of strings or a string."
    ):
        tde.compute(symmetrise=True)
    with pytest.raises(
        ValueError, match="The value of `symmetrise` is not recognised."
    ):
        tde.compute(symmetrise="not_a_symmetrise")

    with pytest.raises(
        TypeError, match="`method` must be a list of ints or an int."
    ):
        tde.compute(method=0.5)
    with pytest.raises(
        ValueError, match="The value of `method` is not recognised."
    ):
        tde.compute(method=0)
    with pytest.raises(
        ValueError, match="The value of `method` is not recognised."
    ):
        tde.compute(method=[1, 5])

    with pytest.raises(TypeError, match="`indices` must be a tuple."):
        tde.compute(indices=list(indices))
    with pytest.raises(ValueError, match="`indices` must have a length of 2."):
        tde.compute(indices=(0, 1, 2))
    with pytest.raises(TypeError, match="Entries of `indices` must be lists."):
        tde.compute(indices=(0, 1))
    with pytest.raises(
        TypeError,
        match="Entries for seeds and targets in `indices` must be ints.",
    ):
        tde.compute(indices=([0.0], [1.0]))
    with pytest.raises(
        ValueError,
        match=(
            "`indices` contains indices for channels not present in the data."
        ),
    ):
        tde.compute(indices=([0], [99]))
    with pytest.raises(
        ValueError, match="Entries of `indices` must have equal length."
    ):
        tde.compute(indices=([0], [1, 2]))
    with pytest.raises(
        ValueError,
        match=(
            "Seeds and targets in `indices` must not be the same channel for "
            "any connection."
        ),
    ):
        tde.compute(indices=([0], [0]))

    with pytest.raises(TypeError, match="`n_jobs` must be an integer."):
        tde.compute(n_jobs=0.5)
    with pytest.raises(ValueError, match="`n_jobs` must be >= 1 or -1."):
        tde.compute(n_jobs=0)


def test_tde_runs() -> None:
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
        return_neg_freqs=True,
    )

    # check it runs with correct inputs
    tde = TDE(data=fft, freqs=freqs, sampling_freq=sampling_freq)
    tde.compute(symmetrise=["none", "antisym"], method=[1, 2, 3, 4])

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

    for symmetrise_arg, symmetrise_name in zip(
        ["none", "antisym"], ["", "(antisymmetrised) "]
    ):
        for method_arg, method_name in zip(
            [1, 2, 3, 4], ["I", "II", "III", "IV"]
        ):
            tde.compute(symmetrise=symmetrise_arg, method=method_arg)
            assert isinstance(tde.results, ResultsTDE)
            assert (
                tde.results.name
                == f"TDE {symmetrise_name}| Method {method_name}"
            )


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
    data = np.load(
        os.path.join(
            "..", "examples", "data", "sim_data_tde_independent_noise.npy"
        )
    )
    sampling_freq = 200  # Hz
    n_times = data.shape[2]

    # compute FFT
    fft_coeffs, freqs = compute_fft(
        data=data,
        sampling_freq=sampling_freq,
        n_points=2 * n_times + 1,
        window="hamming",
        return_neg_freqs=True,
    )

    # compute TDE (seed -> target; target -> seed)
    tde = TDE(data=fft_coeffs, freqs=freqs, sampling_freq=sampling_freq)
    tde.compute(indices=([0, 1], [1, 0]), method=[1, 2, 3, 4])

    # check that 250 ms delay is detected (negative for target -> seed)
    assert (results.tau[0] == tau for results in tde.results)
    assert (results.tau[1] == -tau for results in tde.results)

    # test that TDE is detected at 250 ms with correlated noise (with antisym.)
    # load simulated data with correlated noise
    data = np.load(
        os.path.join(
            "..", "examples", "data", "sim_data_tde_correlated_noise.npy"
        )
    )
    sampling_freq = 200  # Hz
    n_times = data.shape[2]

    # compute FFT
    fft_coeffs, freqs = compute_fft(
        data=data,
        sampling_freq=sampling_freq,
        n_points=2 * n_times + 1,
        window="hamming",
        return_neg_freqs=True,
    )

    # compute TDE
    tde = TDE(data=fft_coeffs, freqs=freqs, sampling_freq=sampling_freq)
    tde.compute(indices=([0], [1]), method=[1, 2, 3, 4], symmetrise="none")
    # check that 0 ms delay is dominant without antisymmetrisation
    assert (results.tau[0] == 0 for results in tde.results)

    tde.compute(indices=([0], [1]), method=[1, 2, 3, 4], symmetrise="antisym")
    # check that 250 ms delay is recovered with antisymmetrisation
    assert (results.tau[0] == tau for results in tde.results)
