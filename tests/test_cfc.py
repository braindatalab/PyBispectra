"""Tests for CFC tools."""

import os

import pytest
import numpy as np

from pybispectra.cfc import AAC, PAC, PPC
from pybispectra.utils import ResultsCFC, compute_fft, compute_tfr
from pybispectra.utils._utils import _generate_data


@pytest.mark.parametrize("class_type", ["PAC", "PPC", "AAC"])
def test_error_catch(class_type: str) -> None:
    """Check that CFC classes catch errors."""
    if class_type == "PAC":
        TestClass = PAC
    elif class_type == "PPC":
        TestClass = PPC
    elif class_type == "AAC":
        TestClass = AAC
    else:
        raise ValueError("`class_type` must be a CFC class.")

    n_chans = 3
    n_epochs = 5
    n_times = 100
    sampling_freq = 50
    data = _generate_data(n_epochs, n_chans, n_times)
    indices = ([0, 1, 2], [0, 1, 2])
    freqs = np.arange(5, 20)

    if class_type in ["PAC", "PPC"]:
        coeffs, freqs = compute_fft(data, sampling_freq)
    else:
        coeffs, freqs = compute_tfr(data, sampling_freq, freqs, n_cycles=3)

    # initialisation
    with pytest.raises(TypeError, match="`data` must be a NumPy array."):
        TestClass(coeffs.tolist(), freqs, sampling_freq)
    if class_type in ["PAC", "PPC"]:
        with pytest.raises(ValueError, match="`data` must be a 3D array."):
            TestClass(np.random.randn(2, 2), freqs, sampling_freq)
    else:
        with pytest.raises(ValueError, match="`data` must be a 4D array."):
            TestClass(np.random.randn(2, 2), freqs, sampling_freq)

    with pytest.raises(TypeError, match="`freqs` must be a NumPy array."):
        TestClass(coeffs, freqs.tolist(), sampling_freq)
    with pytest.raises(ValueError, match="`freqs` must be a 1D array."):
        TestClass(coeffs, np.random.randn(2, 2), sampling_freq)

    with pytest.raises(
        ValueError,
        match=(
            "`data` and `freqs` must contain the same number of frequencies."
        ),
    ):
        TestClass(coeffs, freqs[:-1], sampling_freq)

    with pytest.raises(ValueError, match="Entries of `freqs` must be >= 0."):
        TestClass(coeffs, freqs * -1, sampling_freq)
    with pytest.raises(
        ValueError,
        match=(
            "Entries of `freqs` corresponding to positive frequencies must be "
            "in ascending order."
        ),
    ):
        TestClass(coeffs, freqs[::-1], sampling_freq)

    with pytest.raises(TypeError, match="`verbose` must be a bool."):
        TestClass(coeffs, freqs, sampling_freq, "verbose")

    # compute
    test_class = TestClass(coeffs, freqs, sampling_freq)

    if class_type == "PAC":
        with pytest.raises(
            TypeError,
            match="`symmetrise` must be a list of strings or a string.",
        ):
            test_class.compute(symmetrise=True)
        with pytest.raises(
            ValueError, match="The value of `symmetrise` is not recognised."
        ):
            test_class.compute(symmetrise="not_a_symmetrise")

        with pytest.raises(
            TypeError,
            match="`normalise` must be a list of strings or a string.",
        ):
            test_class.compute(normalise=True)
        with pytest.raises(
            ValueError, match="The value of `normalise` is not recognised."
        ):
            test_class.compute(normalise="not_a_normalise")

    with pytest.raises(TypeError, match="`indices` must be a tuple."):
        test_class.compute(indices=list(indices))
    with pytest.raises(ValueError, match="`indices` must have a length of 2."):
        test_class.compute(indices=(0, 1, 2))
    with pytest.raises(TypeError, match="Entries of `indices` must be lists."):
        test_class.compute(indices=(0, 1))
    with pytest.raises(
        TypeError,
        match="Entries for seeds and targets in `indices` must be ints.",
    ):
        test_class.compute(indices=([0.0], [1.0]))
    with pytest.raises(
        ValueError,
        match=(
            "`indices` contains indices for channels not present in the data."
        ),
    ):
        test_class.compute(indices=([0], [99]))
    with pytest.raises(
        ValueError, match="Entries of `indices` must have equal length."
    ):
        test_class.compute(indices=([0], [1, 2]))

    with pytest.raises(
        TypeError, match="`f1s` and `f2s` must be NumPy arrays."
    ):
        test_class.compute(f1s=freqs.tolist())
    with pytest.raises(
        TypeError, match="`f1s` and `f2s` must be NumPy arrays."
    ):
        test_class.compute(f2s=freqs.tolist())
    with pytest.raises(ValueError, match="`f1s` and `f2s` must be 1D arrays."):
        test_class.compute(f1s=np.random.rand(2, 2))
    with pytest.raises(ValueError, match="`f1s` and `f2s` must be 1D arrays."):
        test_class.compute(f2s=np.random.rand(2, 2))
    with pytest.raises(
        ValueError,
        match=(
            "All frequencies in `f1s` and `f2s` must be present in the data."
        ),
    ):
        test_class.compute(f1s=freqs * -1)
    with pytest.raises(
        ValueError,
        match=(
            "All frequencies in `f1s` and `f2s` must be present in the data."
        ),
    ):
        test_class.compute(f2s=freqs * -1)
    with pytest.raises(
        ValueError, match="Entries of `f1s` must be in ascending order."
    ):
        test_class.compute(f1s=freqs[::-1])
    with pytest.raises(
        ValueError, match="Entries of `f2s` must be in ascending order."
    ):
        test_class.compute(f2s=freqs[::-1])

    with pytest.raises(TypeError, match="`n_jobs` must be an integer."):
        test_class.compute(n_jobs=0.5)
    with pytest.raises(ValueError, match="`n_jobs` must be >= 1 or -1."):
        test_class.compute(n_jobs=0)


def test_pac_runs() -> None:
    """Test that PAC runs correctly."""
    n_chans = 3
    sampling_freq = 50
    data = _generate_data(5, n_chans, 100)

    fft, freqs = compute_fft(
        data=data, sampling_freq=sampling_freq, verbose=False
    )

    # check it runs with correct inputs
    pac = PAC(data=fft, freqs=freqs, sampling_freq=sampling_freq)
    pac.compute(
        symmetrise=["none", "antisym"], normalise=["none", "threenorm"]
    )

    # check the returned results have the correct shape
    assert (
        results.shape == (n_chans**2, len(freqs), len(freqs))
        for results in pac.results
    )

    # check the returned results are of the correct type
    result_types = [
        "PAC | Bispectrum",
        "PAC | Bicoherence",
        "PAC (antisymmetrised) | Bispectrum",
        "PAC (antisymmetrised) | Bicoherence",
    ]
    assert (
        results.name == result_types[i]
        for i, results in enumerate(pac.results)
    )
    assert (isinstance(results, ResultsCFC) for results in pac.results)

    pac.compute(symmetrise="none", normalise="none")
    assert isinstance(pac.results, ResultsCFC)
    assert pac.results.name == result_types[0]

    pac.compute(symmetrise="none", normalise="threenorm")
    assert isinstance(pac.results, ResultsCFC)
    assert pac.results.name == result_types[1]

    pac.compute(symmetrise="antisym", normalise="none")
    assert isinstance(pac.results, ResultsCFC)
    assert pac.results.name == result_types[2]

    pac.compute(symmetrise="antisym", normalise="threenorm")
    assert isinstance(pac.results, ResultsCFC)
    assert pac.results.name == result_types[3]

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


def test_pac_results():
    """Test that PAC returns the correct results.

    Simulated data with 10-60 Hz PAC is used. Bivariate PAC involves genuine
    PAC between channels. Univariate PAC contains genuine PAC only within each
    channel, however this will also appear between channels unless
    antisymmetrisation is used.
    """
    # identify interacting and non-interacting frequencies (10-60 Hz PAC)
    interacting_f1s = np.arange(9, 12)
    interacting_f2s = np.arange(59, 62)
    noninteracting_f1s = np.arange(5, 16)
    noninteracting_f2s = np.arange(55, 66)
    noninteracting_f1s = noninteracting_f1s[
        np.invert(np.isin(noninteracting_f1s, interacting_f1s))
    ]
    noninteracting_f2s = noninteracting_f2s[
        np.invert(np.isin(noninteracting_f2s, interacting_f2s))
    ]

    # test that genuine PAC is detected
    # load simulated data with bivariate PAC interactions
    data = np.load(
        os.path.join("..", "examples", "data", "sim_data_pac_bivariate.npy")
    )
    sampling_freq = 200  # sampling frequency in Hz

    # compute FFT
    fft_coeffs, freqs = compute_fft(
        data=data,
        sampling_freq=sampling_freq,
        n_points=sampling_freq,
    )

    # compute PAC
    pac = PAC(data=fft_coeffs, freqs=freqs, sampling_freq=sampling_freq)
    pac.compute(indices=([0], [1]), symmetrise="none", normalise="none")
    results = pac.results.get_results()

    # check that 10-60 Hz PAC is detected
    assert (
        results[0][np.ix_(interacting_f1s, interacting_f2s)].mean()
        > results[0][np.ix_(noninteracting_f1s, noninteracting_f2s)].mean()
    )

    # Test that spurious PAC is corrected with antisymmetrisation
    # load simulated data with univariate PAC interactions
    data = np.load(
        os.path.join("..", "examples", "data", "sim_data_pac_univariate.npy")
    )
    sampling_freq = 200  # sampling frequency in Hz

    # compute FFT
    fft_coeffs, freqs = compute_fft(
        data=data,
        sampling_freq=sampling_freq,
        n_points=sampling_freq,
    )

    # compute PAC
    pac = PAC(data=fft_coeffs, freqs=freqs, sampling_freq=sampling_freq)
    pac.compute(
        indices=([0, 1, 0], [0, 1, 1]), symmetrise="none", normalise="none"
    )
    results = pac.results.get_results()

    # check that 10-60 Hz PAC is detected within each channel
    assert (
        results[0][np.ix_(interacting_f1s, interacting_f2s)].mean()
        > results[0][np.ix_(noninteracting_f1s, noninteracting_f2s)].mean()
    )  # seed = 0; target = 0
    assert (
        results[1][np.ix_(interacting_f1s, interacting_f2s)].mean()
        > results[1][np.ix_(noninteracting_f1s, noninteracting_f2s)].mean()
    )  # seed = 1; target = 1

    # check that spurious 10-60 Hz PAC is detected between the channels
    assert (
        results[2][np.ix_(interacting_f1s, interacting_f2s)].mean()
        > results[2][np.ix_(noninteracting_f1s, noninteracting_f2s)].mean()
    )  # seed = 0; target = 1

    # check that spurious PAC across channels is removed with antisym.
    pac.compute(indices=([0], [1]), symmetrise="antisym", normalise="none")
    results = pac.results.get_results()
    assert np.isclose(
        results[0][np.ix_(interacting_f1s, interacting_f2s)].mean(),
        results[0][np.ix_(noninteracting_f1s, noninteracting_f2s)].mean(),
        atol=1e-4,
    )


def test_ppc_runs() -> None:
    """Test that PPC runs correctly."""
    n_chans = 3
    sampling_freq = 50
    data = _generate_data(5, n_chans, 100)

    fft, freqs = compute_fft(
        data=data, sampling_freq=sampling_freq, verbose=False
    )

    # check it runs with correct inputs
    ppc = PPC(data=fft, freqs=freqs, sampling_freq=sampling_freq)
    ppc.compute()

    # check the returned results have the correct shape
    assert ppc.results.shape == (n_chans**2, len(freqs), len(freqs))

    # check the returned results are of the correct type
    assert ppc.results.name == "PPC"
    assert isinstance(ppc.results, ResultsCFC)


def test_aac_runs() -> None:
    """Test that AAC runs correctly."""
    n_chans = 3
    sampling_freq = 50
    data = _generate_data(5, n_chans, 100)
    freqs = np.arange(5, 20)

    fft, freqs = compute_tfr(data, sampling_freq, freqs, n_cycles=3)

    # check it runs with correct inputs
    aac = AAC(data=fft, freqs=freqs, sampling_freq=sampling_freq)
    aac.compute()

    # check the returned results have the correct shape
    assert aac.results.shape == (n_chans**2, len(freqs), len(freqs))

    # check the returned results are of the correct type
    assert aac.results.name == "AAC"
    assert isinstance(aac.results, ResultsCFC)
