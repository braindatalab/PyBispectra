"""Tests for general bispectrum and threenorm tools."""

import numpy as np
import pytest
from numpy.random import RandomState

from pybispectra.cfc import PAC
from pybispectra.general import Bispectrum, Threenorm
from pybispectra.utils import (
    ResultsGeneral,
    compute_fft,
    compute_tfr,
    get_example_data_paths,
)
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
    data = _generate_data((n_epochs, n_chans, n_times))
    indices = ([0, 1, 2], [0, 1, 2])
    freqs = np.arange(5, 20)

    coeffs, freqs = compute_fft(data, sampling_freq)

    # initialisation
    with pytest.raises(TypeError, match="`data` must be a NumPy array."):
        TestClass(coeffs.tolist(), freqs, sampling_freq)
    with pytest.raises(ValueError, match="`data` must be a 3D or 4D array."):
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
        TestClass(coeffs, freqs, sampling_freq, verbose="verbose")

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


@pytest.mark.parametrize("class_type", ["Bispectrum", "Threenorm"])
def test_error_catch_time_resolved(class_type: str) -> None:
    """Check that General classes catch errors for time-resolved data."""
    if class_type == "Bispectrum":
        TestClass = Bispectrum
    else:
        TestClass = Threenorm

    n_chans = 3
    n_epochs = 5
    n_times = 100
    sampling_freq = 50
    data = _generate_data((n_epochs, n_chans, n_times))
    freqs = np.arange(5, 20)
    times = np.arange(n_times) / sampling_freq

    coeffs, freqs = compute_tfr(
        data, sampling_freq, freqs, n_cycles=3, output="complex"
    )

    # initialisation
    with pytest.raises(ValueError, match="`data` must be a 3D or 4D array."):
        TestClass(np.random.randn(2, 2), freqs, sampling_freq)

    with pytest.raises(TypeError, match="`times` must be a NumPy array."):
        TestClass(coeffs, freqs, sampling_freq, times.tolist())
    with pytest.raises(ValueError, match="`times` must be a 1D array."):
        TestClass(coeffs, freqs, sampling_freq, times[:, np.newaxis])

    with pytest.raises(
        ValueError,
        match=("`data` and `times` must contain the same number of timepoints."),
    ):
        TestClass(coeffs, freqs, sampling_freq, times[:-1])

    # compute
    test_class = TestClass(coeffs, freqs, sampling_freq, times)

    # test that errors for incorrect inputs are caught
    with pytest.raises(TypeError, match="`times` must be a tuple or None."):
        test_class.compute(times=[times[0], times[-1]])
    with pytest.raises(ValueError, match="`times` must have length of 2."):
        test_class.compute(times=(times[0], times[1], times[-1]))
    with pytest.raises(TypeError, match="Entries of `times` must be int or float."):
        test_class.compute(times=("start", "end"))
    with pytest.raises(
        ValueError,
        match="No timepoints are present in the data for the range in `times`.",
    ):
        test_class.compute(times=(-1, -0.1))


@pytest.mark.parametrize("class_type", ["Bispectrum", "Threenorm"])
def test_general_runs(class_type: str) -> None:
    """Test that General classes run correctly."""
    if class_type == "Bispectrum":
        TestClass = Bispectrum
    else:
        TestClass = Threenorm

    n_chans = 3
    n_times = 100
    sampling_freq = 50
    data = _generate_data((5, n_chans, n_times))
    default_times = np.arange(n_times) / sampling_freq  # matches auto-generated times
    times = default_times + 10  # offset to distinguish from auto-generated ones
    freqs = np.arange(5, 25, 0.5)

    fft, fft_freqs = compute_fft(data=data, sampling_freq=sampling_freq, verbose=False)
    fft = fft[..., np.intersect1d(fft_freqs, freqs, return_indices=True)[1]]
    tfr, _ = compute_tfr(
        data=data,
        sampling_freq=sampling_freq,
        freqs=freqs,
        n_cycles=3,
        output="complex",
    )

    # check data is stored correctly
    test_class = TestClass(data=fft, freqs=freqs, sampling_freq=sampling_freq)
    assert np.all(test_class.data == fft), "FFT data not stored correctly"
    test_class_tr = TestClass(data=tfr, freqs=freqs, sampling_freq=sampling_freq)
    assert np.all(test_class_tr.data == tfr), "TFR data not stored correctly"

    # check times are handled correctly
    test_class = TestClass(
        data=fft, freqs=freqs, sampling_freq=sampling_freq, times=times
    )
    assert test_class.times is None, (
        "`times` should be ignored for non-time-resolved_data"
    )
    test_class = TestClass(
        data=tfr, freqs=freqs, sampling_freq=sampling_freq, times=times
    )
    assert np.all(test_class.times == times), (
        "`times` should be stored for time-resolved_data"
    )
    test_class = TestClass(data=tfr, freqs=freqs, sampling_freq=sampling_freq)
    assert np.all(test_class.times == default_times), (
        "Auto-generated `times` are incorrect for time-resolved_data"
    )

    # check it runs with correct inputs
    test_class = TestClass(data=fft, freqs=freqs, sampling_freq=sampling_freq)
    test_class.compute()
    test_class_tr = TestClass(
        data=tfr, freqs=freqs, sampling_freq=sampling_freq, times=times
    )
    test_class_tr.compute()

    # check the returned results have the correct shape
    assert test_class.results.shape == (n_chans**3, len(freqs), len(freqs))
    assert test_class_tr.results.shape == (
        n_chans**3,
        len(freqs),
        len(freqs),
        len(times),
    )

    # check the returned results are of the correct type
    assert isinstance(test_class.results, ResultsGeneral)
    assert test_class.results.name == class_type

    # check it runs with non-exact frequencies
    fmin, fmax = 10.25, 19.75
    freqs_sel = freqs[np.argwhere((freqs >= fmin) & (freqs <= fmax)).squeeze()]
    test_class.compute(f1s=(fmin, fmax), f2s=(fmin, fmax))
    assert test_class.results.get_results().shape[1:3] == (
        len(freqs_sel),
        len(freqs_sel),
    ), "Number of frequencies in results does not match the selection"
    assert (
        test_class.results.f1s[0] == freqs_sel[0]
        and test_class.results.f1s[-1] == freqs_sel[-1]
        and test_class.results.f2s[0] == freqs_sel[0]
        and test_class.results.f2s[-1] == freqs_sel[-1]
    ), "`f1s` and `f2s` in results do not match the selection"

    # check it runs with non-exact times
    tmin, tmax = 10.55, 11.55
    times_sel = times[np.argwhere((times >= tmin) & (times <= tmax)).squeeze()]
    test_class_tr.compute(times=(tmin, tmax))
    assert test_class_tr.results.get_results().shape[3] == len(times_sel), (
        "Number of timepoints in results does not match the selection"
    )
    assert (
        test_class_tr.results.times[0] == times_sel[0]
        and test_class_tr.results.times[-1] == times_sel[-1]
    ), "`times` in results do not match the selection"

    # test it runs with parallelisation
    test_class.compute(n_jobs=2)
    test_class.compute(n_jobs=-1)

    # test copying works
    test_class_copy = test_class.copy()
    attrs = test_class.__dict__.keys()
    for attr in attrs:
        if not attr.startswith("_"):
            assert np.all(getattr(test_class, attr) == getattr(test_class_copy, attr))
    assert test_class is not test_class_copy


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
