"""Tests for toolbox utilities."""

import pytest
import numpy as np

from pybispectra import compute_fft, fast_find_first, _generate_data


def test_compute_fft() -> None:
    """Test `compute_fft`."""
    n_epochs = 5
    n_chans = 3
    n_times = 100
    data = generate_data(n_epochs, n_chans, n_times)
    sfreq = 50

    # check it runs with correct inputs
    fft, freqs = compute_fft(data=data, sfreq=sfreq, n_jobs=1)
    assert isinstance(fft, np.ndarray), "`fft` should be a NumPy array."
    assert fft.ndim == 3, "`fft` should have 3 dimensions."
    assert fft.shape[:1] == (n_epochs, n_chans), (
        "The first 2 dimensions of `fft` should have shape [epochs x "
        "channels]."
    )
    assert isinstance(freqs, np.ndarray), "`freqs` should be a NumPy array."
    assert freqs.ndim == 1, "`freqs` should have 1 dimension."
    assert (
        fft.shape[2] == freqs.shape[0]
    ), "The 3rd dimension of `fft` should have the same length as `freqs`."
    assert (
        freqs[-1] == sfreq / 2
    ), "The maximum of `freqs` should be the Nyquist frequency."
    assert freqs[0] != 0, "The zero frequency should not be included."

    # check it catches incorrect inputs
    with pytest.raises(TypeError, match="`data` must be a NumPy array."):
        compute_fft(data=data.tolist(), sfreq=sfreq)
    with pytest.raises(ValueError, match="`data` must be a 3D array."):
        compute_fft(data=data[..., 0], sfreq=sfreq)

    with pytest.raises(TypeError, match="`n_jobs` must be an integer."):
        compute_fft(data=data, sfreq=sfreq, n_jobs=[])
    with pytest.raises(TypeError, match="`n_jobs` must be >= 1."):
        compute_fft(data=data, sfreq=sfreq, n_jobs=0)

    # check it works with parallelisation (already tested without)
    compute_fft(data=data, sfreq=sfreq, n_jobs=2)

    # check a warning is raised if `data` is complex-valued
    with warnings.catch_warnings():
        compute_fft(data=np.array(data, dtype=np.complex128) + 1j, sfreq=sfreq)


def test_fast_find_first():
    """Test `fast_find_first`."""
    # test that a present value is found
    index = fast_find_first(vector=np.array([-1, 0, 1, 2, 3, 1]), value=1)
    assert index == 2, "The index of the value being found should be 2."

    # test that a missing value is not found
    with pytest.raises(
        ValueError, match="`value` is not present in `vector`."
    ):
        fast_find_first(vector=np.array([-1, 0, 2, 2, 3, 4]), value=1)
