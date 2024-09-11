"""Tests for general bispectrum and threenorm tools."""

import numpy as np
import pytest
from numpy.random import RandomState

from pybispectra.cfc import PAC
from pybispectra.data import get_example_data_paths
from pybispectra.general import Bispectrum, Threenorm
from pybispectra.utils import ResultsGeneral, compute_fft
from pybispectra.utils._utils import _generate_data
from pybispectra.waveshape import WaveShape


@pytest.mark.parametrize("class_type", ["Bispectrum", "Threenorm"])
def test_error_catch(class_type: str) -> None:
    """Check that General classes catch errors."""
    if class_type == "Bispectrum":
        TestClass = Bispectrum
    else:
        TestClass = Threenorm

    n_chans = 3
    n_epochs = 5
    n_times = 100
    sampling_freq = 50
    data = _generate_data(n_epochs, n_chans, n_times)
    indices = ([0, 1, 2], [0, 1, 2])
    freqs = np.arange(5, 20)

    coeffs, freqs = compute_fft(data, sampling_freq)

    # initialisation
    with pytest.raises(TypeError, match="`data` must be a NumPy array."):
        TestClass(coeffs.tolist(), freqs, sampling_freq)
    with pytest.raises(ValueError, match="`data` must be a 3D array."):
        TestClass(np.random.randn(2, 2), freqs, sampling_freq)

    with pytest.raises(TypeError, match="`freqs` must be a NumPy array."):
        TestClass(coeffs, freqs.tolist(), sampling_freq)
    with pytest.raises(ValueError, match="`freqs` must be a 1D array."):
        TestClass(coeffs, np.random.randn(2, 2), sampling_freq)

    with pytest.raises(
        ValueError,
        match=("`data` and `freqs` must contain the same number of frequencies."),
    ):
        TestClass(coeffs, freqs[:-1], sampling_freq)

    with pytest.raises(ValueError, match="Entries of `freqs` must be >= 0."):
        TestClass(coeffs, freqs * -1, sampling_freq)
    with pytest.raises(
        ValueError, match="At least one entry of `freqs` is > the Nyquist frequency."
    ):
        bad_freqs = np.linspace(0, sampling_freq / 2 + 1, freqs.size)
        TestClass(coeffs, bad_freqs, sampling_freq)
    with pytest.raises(
        ValueError, match=("Entries of `freqs` must be in ascending order.")
    ):
        TestClass(coeffs, freqs[::-1], sampling_freq)
    with pytest.raises(ValueError, match="Entries of `freqs` must be evenly spaced."):
        bad_freqs = freqs.copy()
        bad_freqs[1] *= 2
        TestClass(coeffs, bad_freqs, sampling_freq)

    with pytest.raises(TypeError, match="`sampling_freq` must be an int or a float."):
        TestClass(coeffs, freqs, None)

    with pytest.raises(TypeError, match="`verbose` must be a bool."):
        TestClass(coeffs, freqs, sampling_freq, "verbose")

    # compute
    test_class = TestClass(coeffs, freqs, sampling_freq)

    with pytest.raises(TypeError, match="`indices` must be a tuple."):
        test_class.compute(indices=list(indices))
    with pytest.raises(ValueError, match="`indices` must have length of 3."):
        test_class.compute(indices=(0, 1))
    with pytest.raises(TypeError, match="Entries of `indices` must be tuples."):
        test_class.compute(indices=(0, 1, 2))
    with pytest.raises(
        TypeError, match="Entries for groups in `indices` must be ints."
    ):
        test_class.compute(indices=((0.0,), (1.0,), (1.0,)))
    with pytest.raises(
        ValueError,
        match=("`indices` contains indices for channels not present in the data."),
    ):
        test_class.compute(indices=((0,), (1,), (99,)))
    with pytest.raises(
        ValueError, match="Entries of `indices` must have equal length."
    ):
        test_class.compute(indices=((0,), (1,), (0, 1)))

    with pytest.raises(TypeError, match="`f1s` and `f2s` must be tuples."):
        test_class.compute(f1s=[freqs[0], freqs[-1]])
    with pytest.raises(TypeError, match="`f1s` and `f2s` must be tuples."):
        test_class.compute(f2s=[freqs[0], freqs[-1]])
    with pytest.raises(ValueError, match="`f1s` and `f2s` must have lengths of 2."):
        test_class.compute(f1s=(freqs[0], freqs[1], freqs[-1]))
    with pytest.raises(ValueError, match="`f1s` and `f2s` must have lengths of 2."):
        test_class.compute(f2s=(freqs[0], freqs[1], freqs[-1]))
    with pytest.raises(ValueError, match="Entries of `f1s` and `f2s` must be >= 0."):
        test_class.compute(f1s=(-1, 10))
    with pytest.raises(ValueError, match="Entries of `f1s` and `f2s` must be >= 0."):
        test_class.compute(f1s=(10, -1))
    with pytest.raises(ValueError, match="Entries of `f1s` and `f2s` must be >= 0."):
        test_class.compute(f2s=(-1, 10))
    with pytest.raises(ValueError, match="Entries of `f1s` and `f2s` must be >= 0."):
        test_class.compute(f2s=(10, -1))
    with pytest.raises(
        ValueError, match="Entries of `f1s` and `f2s` must be <= the Nyquist frequency."
    ):
        test_class.compute(f1s=(5, sampling_freq / 2 + 1))
    with pytest.raises(
        ValueError, match="Entries of `f1s` and `f2s` must be <= the Nyquist frequency."
    ):
        test_class.compute(f1s=(sampling_freq / 2 + 1, 10))
    with pytest.raises(
        ValueError, match="Entries of `f1s` and `f2s` must be <= the Nyquist frequency."
    ):
        test_class.compute(f2s=(5, sampling_freq / 2 + 1))
    with pytest.raises(
        ValueError, match="Entries of `f1s` and `f2s` must be <= the Nyquist frequency."
    ):
        test_class.compute(f2s=(sampling_freq / 2 + 1, 10))
    with pytest.raises(
        ValueError,
        match="No frequencies are present in the data for the range in `f1s`.",
    ):
        test_class.compute(f1s=(10, 5))
    with pytest.raises(
        ValueError,
        match="No frequencies are present in the data for the range in `f1s`.",
    ):
        test_class.compute(f1s=(5.6, 5.7))
    with pytest.raises(
        ValueError,
        match="No frequencies are present in the data for the range in `f2s`.",
    ):
        test_class.compute(f2s=(10, 5))
    with pytest.raises(
        ValueError,
        match="No frequencies are present in the data for the range in `f2s`.",
    ):
        test_class.compute(f2s=(5.6, 5.7))

    with pytest.raises(TypeError, match="`n_jobs` must be an integer."):
        test_class.compute(n_jobs=0.5)
    with pytest.raises(ValueError, match="`n_jobs` must be >= 1 or -1."):
        test_class.compute(n_jobs=0)


def test_bispectrum_runs() -> None:
    """Test that Bispectrum runs correctly."""
    n_chans = 3
    sampling_freq = 50
    data = _generate_data(5, n_chans, 100)

    fft, freqs = compute_fft(data=data, sampling_freq=sampling_freq, verbose=False)

    # check it runs with correct inputs
    bs = Bispectrum(data=fft, freqs=freqs, sampling_freq=sampling_freq)
    bs.compute()

    # check the returned results have the correct shape
    assert bs.results.shape == (n_chans**3, len(freqs), len(freqs))

    # check the returned results are of the correct type
    assert isinstance(bs.results, ResultsGeneral)
    assert bs.results.name == "Bispectrum"

    # check it runs with non-exact frequencies
    bs.compute(f1s=(10.25, 19.75), f2s=(10.25, 19.75))

    # test it runs with parallelisation
    bs.compute(n_jobs=2)
    bs.compute(n_jobs=-1)

    # test copying works
    bs_copy = bs.copy()
    attrs = bs.__dict__.keys()
    for attr in attrs:
        if not attr.startswith("_"):
            assert np.all(getattr(bs, attr) == getattr(bs_copy, attr))
    assert bs is not bs_copy


def test_threenorm_runs() -> None:
    """Test that Threenorm runs correctly."""
    n_chans = 3
    sampling_freq = 50
    data = _generate_data(5, n_chans, 100)

    fft, freqs = compute_fft(data=data, sampling_freq=sampling_freq, verbose=False)

    # check it runs with correct inputs
    norm = Threenorm(data=fft, freqs=freqs, sampling_freq=sampling_freq)
    norm.compute()

    # check the returned results have the correct shape
    assert norm.results.shape == (n_chans**3, len(freqs), len(freqs))

    # check the returned results are of the correct type
    assert isinstance(norm.results, ResultsGeneral)
    assert norm.results.name == "Threenorm"

    # check it runs with non-exact frequencies
    norm.compute(f1s=(10.25, 19.75), f2s=(10.25, 19.75))

    # test it runs with parallelisation
    norm.compute(n_jobs=2)
    norm.compute(n_jobs=-1)

    # test copying works
    norm_copy = norm.copy()
    attrs = norm.__dict__.keys()
    for attr in attrs:
        if not attr.startswith("_"):
            assert np.all(getattr(norm, attr) == getattr(norm_copy, attr))
    assert norm is not norm_copy


def test_pac_results():
    """Test that PAC returns the correct results.

    Simulated data with 10-60 Hz PAC is used. Bivariate PAC involves genuine PAC between
    channels. Univariate PAC contains genuine PAC only within each channel, however this
    will also appear between channels unless antisymmetrisation is used.
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
    data = np.load(get_example_data_paths("sim_data_pac_bivariate"))
    sampling_freq = 200  # sampling frequency in Hz

    # compute FFT
    fft_coeffs, freqs = compute_fft(
        data=data,
        sampling_freq=sampling_freq,
        n_points=sampling_freq,
    )

    # compute PAC
    pac = Bispectrum(data=fft_coeffs, freqs=freqs, sampling_freq=sampling_freq)
    pac.compute(indices=((0,), (1,), (1,)))
    results = pac.results.get_results()

    # check that results match dedicated class
    pac_dedicated = PAC(data=fft_coeffs, freqs=freqs, sampling_freq=sampling_freq)
    pac_dedicated.compute(indices=((0,), (1,)), antisym=False, norm=False)
    pac_dedicated_results = pac_dedicated.results.get_results()
    assert np.all(
        np.abs(results[~np.isnan(np.abs(results))])
        == pac_dedicated_results[~np.isnan(pac_dedicated_results)]
    )

    # compute threenorm
    norm = Threenorm(data=fft_coeffs, freqs=freqs, sampling_freq=sampling_freq)
    norm.compute(indices=((0,), (1,), (1,)))
    pac_norm = np.abs(results / norm.results.get_results())

    # check that normalised PAC matches dedicated class
    pac_dedicated_norm = PAC(data=fft_coeffs, freqs=freqs, sampling_freq=sampling_freq)
    pac_dedicated_norm.compute(indices=((0,), (1,)), antisym=False, norm=True)
    pac_dedicated_norm_results = pac_dedicated_norm.results.get_results()
    assert np.all(
        pac_norm[~np.isnan(pac_norm)]
        == pac_dedicated_norm_results[~np.isnan(pac_dedicated_norm_results)]
    )

    # Test that spurious PAC is corrected with antisymmetrisation
    # load simulated data with univariate PAC interactions
    data = np.load(get_example_data_paths("sim_data_pac_univariate"))
    sampling_freq = 200  # sampling frequency in Hz

    # compute FFT
    fft_coeffs, freqs = compute_fft(
        data=data, sampling_freq=sampling_freq, n_points=sampling_freq
    )

    # compute PAC
    pac = Bispectrum(data=fft_coeffs, freqs=freqs, sampling_freq=sampling_freq)
    pac.compute(indices=((0, 1), (1, 0), (1, 1)))
    results = pac.results.get_results()
    results = np.abs(results[0] - results[1])

    # check that antisymmetrised PAC matches dedicated class
    pac_dedicated_antisym = PAC(
        data=fft_coeffs, freqs=freqs, sampling_freq=sampling_freq
    )
    pac_dedicated_antisym.compute(indices=((0,), (1,)), antisym=True, norm=False)
    pac_dedicated_antisym_results = pac_dedicated_antisym.results.get_results()
    assert np.all(
        results[~np.isnan(results)]
        == pac_dedicated_antisym_results[0][~np.isnan(pac_dedicated_antisym_results[0])]
    )


def test_waveshape_results():
    """Test that WaveShape returns the correct results.

    Simulated data with 10 Hz (plus harmonics) wave shape features is used. Waveshape
    features include a ramp up sawtooth, ramp down sawtooth, peak-dominant signal, and a
    trough-dominant signal.
    """
    all_freqs = (0, 35)

    # test that genuine PAC is detected
    # load simulated data with bivariate PAC interactions
    data_sawtooths = np.load(get_example_data_paths("sim_data_waveshape_sawtooths"))
    data_peaks_troughs = np.load(
        get_example_data_paths("sim_data_waveshape_peaks_troughs")
    )
    sampling_freq = 1000  # sampling frequency in Hz

    # add noise for numerical stability
    random = RandomState(44)
    snr = 0.25
    datasets = [data_sawtooths, data_peaks_troughs]
    for data_idx, data in enumerate(datasets):
        datasets[data_idx] = snr * data + (1 - snr) * random.rand(*data.shape)
    data_sawtooths = datasets[0]
    data_peaks_troughs = datasets[1]

    # compute FFT
    coeffs_sawtooths, freqs = compute_fft(
        data=data_sawtooths, sampling_freq=sampling_freq, n_points=sampling_freq
    )
    coeffs_peaks_troughs, freqs = compute_fft(
        data=data_peaks_troughs, sampling_freq=sampling_freq, n_points=sampling_freq
    )

    # sawtooth waves
    waveshape_sawtooths = Bispectrum(
        data=coeffs_sawtooths, freqs=freqs, sampling_freq=sampling_freq
    )
    waveshape_sawtooths.compute(
        indices=((0, 1), (0, 1), (0, 1)), f1s=all_freqs, f2s=all_freqs
    )
    waveshape_sawtooths_norm = Threenorm(
        data=coeffs_sawtooths, freqs=freqs, sampling_freq=sampling_freq
    )
    waveshape_sawtooths_norm.compute(
        indices=((0, 1), (0, 1), (0, 1)), f1s=all_freqs, f2s=all_freqs
    )
    results_sawtooths = (
        waveshape_sawtooths.results.get_results()
        / waveshape_sawtooths_norm.results.get_results()
    )

    # peaks and troughs
    waveshape_peaks_troughs = Bispectrum(
        data=coeffs_peaks_troughs, freqs=freqs, sampling_freq=sampling_freq
    )
    waveshape_peaks_troughs.compute(
        indices=((0, 1), (0, 1), (0, 1)), f1s=all_freqs, f2s=all_freqs
    )
    waveshape_peaks_troughs_norm = Threenorm(
        data=coeffs_peaks_troughs, freqs=freqs, sampling_freq=sampling_freq
    )
    waveshape_peaks_troughs_norm.compute(
        indices=((0, 1), (0, 1), (0, 1)), f1s=all_freqs, f2s=all_freqs
    )
    results_peaks_troughs = (
        waveshape_peaks_troughs.results.get_results()
        / waveshape_peaks_troughs_norm.results.get_results()
    )

    # check that results match dedicated class
    # sawtooth waves
    waveshape_sawtooths_dedicated = WaveShape(
        data=coeffs_sawtooths, freqs=freqs, sampling_freq=sampling_freq
    )
    waveshape_sawtooths_dedicated.compute(f1s=all_freqs, f2s=all_freqs)
    results_sawtooths_dedicated = waveshape_sawtooths_dedicated.results.get_results()
    assert np.all(
        results_sawtooths[~np.isnan(results_sawtooths)]
        == results_sawtooths_dedicated[~np.isnan(results_sawtooths_dedicated)]
    )

    # peaks and troughs
    waveshape_peaks_troughs_dedicated = WaveShape(
        data=coeffs_peaks_troughs, freqs=freqs, sampling_freq=sampling_freq
    )
    waveshape_peaks_troughs_dedicated.compute(f1s=all_freqs, f2s=all_freqs)
    results_peaks_troughs_dedicated = (
        waveshape_peaks_troughs_dedicated.results.get_results()
    )
    assert np.all(
        results_peaks_troughs[~np.isnan(results_peaks_troughs)]
        == results_peaks_troughs_dedicated[~np.isnan(results_peaks_troughs_dedicated)]
    )
