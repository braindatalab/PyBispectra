"""Pytest fixtures for unit tests."""

import pytest
import numpy as np

from pybispectra import compute_fft, compute_tfr
from pybispectra.utils._utils import _generate_data


@pytest.fixture(scope="session")
def epochs() -> np.ndarray:
    """Real-valued epoched timeseries data."""
    n_epochs, n_chans, n_times = 4, 3, 100
    return _generate_data((n_epochs, n_chans, n_times), complexobj=False)


@pytest.fixture(scope="session")
def data_sfreq() -> float:
    """Sampling frequency of example data for tests."""
    return 50.0


@pytest.fixture(scope="session")
def fft_and_freqs(
    epochs: np.ndarray, data_sfreq: float
) -> tuple[np.ndarray, np.ndarray]:
    """FFT coefficients and frequencies of epoched timeseries data."""
    fft, freqs = compute_fft(epochs, data_sfreq)
    return (fft, freqs)


@pytest.fixture(scope="session")
def power_tfr_and_freqs(
    epochs: np.ndarray, data_sfreq: float
) -> tuple[np.ndarray, np.ndarray]:
    """Power TFR coefficients and frequencies of epoched timeseries data."""
    freqs = np.arange(5, 20)
    tfr, freqs = compute_tfr(epochs, data_sfreq, freqs, n_cycles=3, output="power")
    return (tfr, freqs)


@pytest.fixture(scope="session")
def complex_tfr_and_freqs(
    epochs: np.ndarray, data_sfreq: float
) -> tuple[np.ndarray, np.ndarray]:
    """Complex TFR coefficients and frequencies of epoched timeseries data."""
    freqs = np.arange(5, 20)
    tfr, freqs = compute_tfr(epochs, data_sfreq, freqs, n_cycles=3, output="complex")
    return (tfr, freqs)
