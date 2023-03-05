"""Tests for CFC tools."""

import warnings

import pytest
import numpy as np

from pybispectra.cfc import PAC, PPC
from pybispectra.utils import compute_fft, _generate_data


def test_ppc() -> None:
    """Test PPC."""
    n_chans = 3
    data = _generate_data(5, n_chans, 100)

    fft, freqs = compute_fft(data=data, sfreq=50)

    # check it runs with correct inputs
    ppc = PPC(data=fft, freqs=freqs)
    ppc.compute()
