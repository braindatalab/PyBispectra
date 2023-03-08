"""Tests for CFC tools."""

import warnings

import pytest
import numpy as np

from pybispectra.cfc import PAC, PPC
from pybispectra.utils import Results, compute_fft, _generate_data


def test_ppc() -> None:
    """Test PPC."""
    n_chans = 3
    data = _generate_data(5, n_chans, 100)

    fft, freqs = compute_fft(data=data, sfreq=50, verbose=False)

    # check it runs with correct inputs
    ppc = PPC(data=fft, freqs=freqs)
    ppc.compute()
    ppc.compute(f1=freqs[:-1])

    ppc_copy = ppc.copy()
    del ppc_copy

    # check it catches incorrect inputs
    with pytest.raises(TypeError, match="`data` must be a NumPy array."):
        PPC(data=fft.tolist(), freqs=freqs)
    with pytest.raises(ValueError, match="`data` must be a 3D array."):
        PPC(data=np.random.rand(2, 2), freqs=freqs)

    with pytest.raises(TypeError, match="`freqs` must be a NumPy array."):
        PPC(data=fft, freqs=freqs.tolist())
    with pytest.raises(ValueError, match="`freqs` must be a 1D array."):
        PPC(data=fft, freqs=np.random.rand(2, 2))

    with pytest.raises(
        ValueError,
        match="`data` and `freqs` should contain the same number of",
    ):
        PPC(data=fft[:, :, 1:], freqs=freqs)

    with pytest.raises(TypeError, match="`indices` should be a tuple."):
        ppc.compute(indices=[0, 1])
    with pytest.raises(
        ValueError, match="`indices` should have a length of 2."
    ):
        ppc.compute(indices=(0, 1, 2))
    with pytest.raises(
        TypeError, match="Entries of `indices` should be NumPy arrays."
    ):
        ppc.compute(indices=(0, 1))
    with pytest.raises(
        ValueError,
        match="`indices` contains indices for channels not present in the",
    ):
        ppc.compute(indices=(np.array([-1]), np.array([0])))
    with pytest.raises(
        ValueError, match="Entires of `indices` must have equal length."
    ):
        ppc.compute(indices=(np.array([0]), np.array([1, 2])))

    with pytest.raises(
        TypeError, match="`f1` and `f2` must be NumPy ndarrays."
    ):
        ppc.compute(f1=freqs[:-1].tolist())
    with pytest.raises(
        TypeError, match="`f1` and `f2` must be NumPy ndarrays."
    ):
        ppc.compute(f2=freqs[1:].tolist())

    with pytest.raises(ValueError, match="`f1` and `f2` must be 1D arrays."):
        ppc.compute(f1=np.random.rand(2, 2))
    with pytest.raises(ValueError, match="`f1` and `f2` must be 1D arrays."):
        ppc.compute(f2=np.random.rand(2, 2))

    with pytest.raises(
        ValueError,
        match="All frequencies in `f1` and `f2` must be present in the data.",
    ):
        ppc.compute(f1=freqs[1:] + 10)
    with pytest.raises(
        ValueError,
        match="All frequencies in `f1` and `f2` must be present in the data.",
    ):
        ppc.compute(f2=freqs[:-1] + 10)

    with pytest.raises(TypeError, match="`n_jobs` must be an integer."):
        ppc.compute(n_jobs=0.5)
    with pytest.raises(ValueError, match="`n_jobs` must be >= 1."):
        ppc.compute(n_jobs=0)

    # check a warning is raised if f1 and f2 overlap
    with warnings.catch_warnings():
        ppc.compute()


def test_pac() -> None:
    """Test PAC."""
    n_chans = 3
    data = _generate_data(5, n_chans, 100)

    fft, freqs = compute_fft(data=data, sfreq=50, verbose=False)

    # check it runs with correct inputs
    pac = PAC(data=fft, freqs=freqs)
    pac.compute()

    # check the returned results are of the correct type
    result_types = [
        "PAC - Unsymmetrised Bispectra",
        "PAC - Unsymmetrised Bicoherence",
        "PAC - Antisymmetrised Bispectra",
        "PAC - Antisymmetrised Bicoherence",
    ]
    assert (pac.results[i].name == result_types[i] for i in range(4))

    pac.compute(symmetrise="none", normalise="none")
    assert isinstance(pac.results, Results)
    assert pac.results.name == result_types[0]

    pac.compute(symmetrise="antisym", normalise="none")
    assert isinstance(pac.results, Results)
    assert pac.results.name == result_types[2]

    pac.compute(symmetrise="none", normalise="threenorm")
    assert isinstance(pac.results, Results)
    assert pac.results.name == result_types[1]

    pac.compute(symmetrise=["none", "antisym"], normalise="none")
    assert len(pac.results) == 2
    assert (
        pac.results[i].name == result_types[type_i]
        for i, type_i in enumerate((0, 2))
    )

    pac.compute(symmetrise="none", normalise=["none", "threenorm"])
    assert len(pac.results) == 2
    assert (
        pac.results[i].name == result_types[type_i]
        for i, type_i in enumerate((0, 3))
    )

    # check it catches incorrect inputs
    with pytest.raises(
        TypeError, match="`symmetrise` must be a list of strings or a string."
    ):
        pac.compute(symmetrise=0)
    with pytest.raises(
        TypeError, match="`normalise` must be a list of strings or a string."
    ):
        pac.compute(normalise=0)

    with pytest.raises(
        ValueError, match="The value of `symmetrise` is not recognised."
    ):
        pac.compute(symmetrise="notaform")
    with pytest.raises(
        ValueError, match="The value of `normalise` is not recognised."
    ):
        pac.compute(normalise="notaform")


# test_ppc()
