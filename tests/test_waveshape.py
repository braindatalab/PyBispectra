"""Tests for wave shape tools."""

import numpy as np
import pytest
from numpy.random import RandomState

from pybispectra.utils import (
    ResultsWaveShape,
    compute_fft,
    compute_tfr,
    get_example_data_paths,
)
from pybispectra.utils._utils import _generate_data
from pybispectra.waveshape import WaveShape


def test_error_catch() -> None:
    """Check that WaveShape class catches errors."""
    n_chans = 3
    n_epochs = 5
    n_times = 100
    sampling_freq = 50
    data = _generate_data((n_epochs, n_chans, n_times))
    indices = (0, 1, 2)

    coeffs, freqs = compute_fft(data, sampling_freq)

    # initialisation
    with pytest.raises(TypeError, match="`data` must be a NumPy array."):
        WaveShape(coeffs.tolist(), freqs, sampling_freq)
    with pytest.raises(ValueError, match="`data` must be a 3D or 4D array."):
        WaveShape(np.random.randn(2, 2), freqs, sampling_freq)

    with pytest.raises(TypeError, match="`freqs` must be a NumPy array."):
        WaveShape(coeffs, freqs.tolist(), sampling_freq)
    with pytest.raises(ValueError, match="`freqs` must be a 1D array."):
        WaveShape(coeffs, np.random.randn(2, 2), sampling_freq)

    with pytest.raises(
        ValueError,
        match=("`data` and `freqs` must contain the same number of frequencies."),
    ):
        WaveShape(coeffs, freqs[:-1], sampling_freq)

    with pytest.raises(ValueError, match="Entries of `freqs` must be >= 0."):
        WaveShape(coeffs, freqs * -1, sampling_freq)
    with pytest.raises(
        ValueError, match="At least one entry of `freqs` is > the Nyquist frequency."
    ):
        bad_freqs = np.linspace(0, sampling_freq / 2 + 1, freqs.size)
        WaveShape(coeffs, bad_freqs, sampling_freq)
    with pytest.raises(
        ValueError, match="Entries of `freqs` must be in ascending order."
    ):
        WaveShape(coeffs, freqs[::-1], sampling_freq)
    with pytest.raises(ValueError, match="Entries of `freqs` must be evenly spaced."):
        bad_freqs = freqs.copy()
        bad_freqs[1] *= 2
        WaveShape(coeffs, bad_freqs, sampling_freq)

    with pytest.raises(TypeError, match="`sampling_freq` must be an int or a float."):
        WaveShape(coeffs, freqs, None)

    with pytest.raises(TypeError, match="`verbose` must be a bool."):
        WaveShape(coeffs, freqs, sampling_freq, verbose="verbose")

    # compute
    waveshape = WaveShape(coeffs, freqs, sampling_freq)

    with pytest.raises(TypeError, match="`norm` must be a bool or tuple of bools."):
        waveshape.compute(norm="true")
    with pytest.raises(TypeError, match="Entries of `norm` must be bools."):
        waveshape.compute(norm=("true",))

    with pytest.raises(TypeError, match="`indices` must be a tuple."):
        waveshape.compute(indices=list(indices))
    with pytest.raises(TypeError, match="Entries of `indices` must be ints."):
        waveshape.compute(indices=(0.0, 1.0))
    with pytest.raises(
        ValueError,
        match=("`indices` contains indices for channels not present in the data."),
    ):
        waveshape.compute(indices=(0, 99))

    with pytest.raises(TypeError, match="`f1s` and `f2s` must be tuples."):
        waveshape.compute(f1s=[freqs[0], freqs[-1]])
    with pytest.raises(TypeError, match="`f1s` and `f2s` must be tuples."):
        waveshape.compute(f2s=[freqs[0], freqs[-1]])

    with pytest.raises(TypeError, match="`n_jobs` must be an integer."):
        waveshape.compute(n_jobs=0.5)
    with pytest.raises(ValueError, match="`n_jobs` must be >= 1 or -1."):
        waveshape.compute(n_jobs=0)


def test_error_catch_time_resolved() -> None:
    """Check that WaveShape class catches errors for time-resolved data."""
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
        WaveShape(np.random.randn(2, 2), freqs, sampling_freq)

    with pytest.raises(TypeError, match="`times` must be a NumPy array."):
        WaveShape(coeffs, freqs, sampling_freq, times.tolist())
    with pytest.raises(ValueError, match="`times` must be a 1D array."):
        WaveShape(coeffs, freqs, sampling_freq, times[:, np.newaxis])

    with pytest.raises(
        ValueError,
        match=("`data` and `times` must contain the same number of timepoints."),
    ):
        WaveShape(coeffs, freqs, sampling_freq, times[:-1])

    # compute
    waveshape = WaveShape(coeffs, freqs, sampling_freq, times)

    # test that errors for incorrect inputs are caught
    with pytest.raises(TypeError, match="`times` must be a tuple or None."):
        waveshape.compute(times=[times[0], times[-1]])
    with pytest.raises(ValueError, match="`times` must have length of 2."):
        waveshape.compute(times=(times[0], times[1], times[-1]))
    with pytest.raises(TypeError, match="Entries of `times` must be int or float."):
        waveshape.compute(times=("start", "end"))
    with pytest.raises(
        ValueError,
        match="No timepoints are present in the data for the range in `times`.",
    ):
        waveshape.compute(times=(-1, -0.1))


def test_waveshape_runs() -> None:
    """Test that WaveShape runs correctly."""
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
    waveshape = WaveShape(data=fft, freqs=freqs, sampling_freq=sampling_freq)
    assert np.all(waveshape.data == fft), "FFT data not stored correctly"
    waveshape_tr = WaveShape(data=tfr, freqs=freqs, sampling_freq=sampling_freq)
    assert np.all(waveshape_tr.data == tfr), "TFR data not stored correctly"

    # check times are handled correctly
    waveshape = WaveShape(
        data=fft, freqs=freqs, sampling_freq=sampling_freq, times=times
    )
    assert waveshape.times is None, (
        "`times` should be ignored for non-time-resolved_data"
    )
    waveshape = WaveShape(
        data=tfr, freqs=freqs, sampling_freq=sampling_freq, times=times
    )
    assert np.all(waveshape.times == times), (
        "`times` should be stored for time-resolved_data"
    )
    waveshape = WaveShape(data=tfr, freqs=freqs, sampling_freq=sampling_freq)
    assert np.all(waveshape.times == default_times), (
        "Auto-generated `times` are incorrect for time-resolved_data"
    )

    # check it runs with correct inputs
    waveshape = WaveShape(data=fft, freqs=freqs, sampling_freq=sampling_freq)
    waveshape.compute(norm=(False, True))
    waveshape_tr = WaveShape(
        data=tfr, freqs=freqs, sampling_freq=sampling_freq, times=times
    )
    waveshape_tr.compute()

    # check the returned results have the correct shape
    assert (
        results.shape == (n_chans, len(freqs), len(freqs))
        for results in waveshape.results
    )
    assert waveshape_tr.results.shape == (n_chans, len(freqs), len(freqs), len(times))

    # check the returned results are of the correct type
    result_types = ["Waveshape | Bispectrum", "Waveshape | Bicoherence"]
    assert (
        results.name == result_types[i] for i, results in enumerate(waveshape.results)
    )
    assert (isinstance(results, ResultsWaveShape) for results in waveshape.results)

    waveshape.compute(norm=False)
    assert isinstance(waveshape.results, ResultsWaveShape)
    assert waveshape.results.name == result_types[0]

    waveshape.compute(norm=True)
    assert isinstance(waveshape.results, ResultsWaveShape)
    assert waveshape.results.name == result_types[1]

    # check it runs with non-exact frequencies
    fmin, fmax = 10.25, 19.75
    freqs_sel = freqs[np.argwhere((freqs >= fmin) & (freqs <= fmax)).squeeze()]
    waveshape.compute(f1s=(fmin, fmax), f2s=(fmin, fmax))
    assert waveshape.results.get_results().shape[1:3] == (
        len(freqs_sel),
        len(freqs_sel),
    ), "Number of frequencies in results does not match the selection"
    assert (
        waveshape.results.f1s[0] == freqs_sel[0]
        and waveshape.results.f1s[-1] == freqs_sel[-1]
        and waveshape.results.f2s[0] == freqs_sel[0]
        and waveshape.results.f2s[-1] == freqs_sel[-1]
    ), "`f1s` and `f2s` in results do not match the selection"

    # check it runs with non-exact times
    tmin, tmax = 10.55, 11.55
    times_sel = times[np.argwhere((times >= tmin) & (times <= tmax)).squeeze()]
    waveshape_tr.compute(times=(tmin, tmax))
    assert waveshape_tr.results.get_results().shape[3] == len(times_sel), (
        "Number of timepoints in results does not match the selection"
    )
    assert (
        waveshape_tr.results.times[0] == times_sel[0]
        and waveshape_tr.results.times[-1] == times_sel[-1]
    ), "`times` in results do not match the selection"

    # test it runs with parallelisation
    waveshape.compute(n_jobs=2)
    waveshape.compute(n_jobs=-1)

    # test copying works
    waveshape_copy = waveshape.copy()
    attrs = waveshape.__dict__.keys()
    for attr in attrs:
        if not attr.startswith("_"):
            assert np.all(getattr(waveshape, attr) == getattr(waveshape_copy, attr))
    assert waveshape is not waveshape_copy


def test_waveshape_results():
    """Test that WaveShape returns the correct results.

    Simulated data with 10 Hz (plus harmonics) wave shape features is used. Waveshape
    features include a ramp up sawtooth, ramp down sawtooth, peak-dominant signal, and
    a trough-dominant signal.
    """
    # tolerance for "closeness"
    close_atol = 0.1

    # identify frequencies of wave shape features (~10 Hz and harmonics)
    focal_freqs = np.array([10, 20, 30])
    all_freqs = (0, 35)

    # load simulated data with non-sinusoidal features
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
    waveshape_sawtooths = WaveShape(
        data=coeffs_sawtooths, freqs=freqs, sampling_freq=sampling_freq
    )
    waveshape_sawtooths.compute(f1s=all_freqs, f2s=all_freqs)
    results_sawtooths = waveshape_sawtooths.results.get_results()

    # peaks and troughs
    waveshape_peaks_troughs = WaveShape(
        data=coeffs_peaks_troughs, freqs=freqs, sampling_freq=sampling_freq
    )
    waveshape_peaks_troughs.compute(f1s=all_freqs, f2s=all_freqs)
    results_peaks_troughs = waveshape_peaks_troughs.results.get_results()

    # check that sawtooth features are detected
    for results, real_val, imag_val, phase in zip(
        results_sawtooths, [0, 0], [1, -1], [0.5, 1.5]
    ):
        assert np.isclose(
            np.nanmean(np.real(results[np.ix_(focal_freqs, focal_freqs)])),
            real_val,
            atol=close_atol,
        )
        assert np.isclose(
            np.nanmean(np.imag(results[np.ix_(focal_freqs, focal_freqs)])),
            imag_val,
            atol=close_atol,
        )
        # normalise phase to [0, 2) in units of pi
        phases = np.angle(results[np.ix_(focal_freqs, focal_freqs)])
        phases[phases < 0] += 2 * np.pi
        phases /= np.pi
        assert np.isclose(np.nanmean(phases), phase, atol=close_atol)

    # check that peak/trough features are detected
    for results, real_val, imag_val, phase in zip(
        results_peaks_troughs, [1, -1], [0, 0], [0, 1]
    ):
        assert np.isclose(
            np.nanmean(np.real(results[np.ix_(focal_freqs, focal_freqs)])),
            real_val,
            atol=close_atol,
        )
        assert np.isclose(
            np.nanmean(np.imag(results[np.ix_(focal_freqs, focal_freqs)])),
            imag_val,
            atol=close_atol,
        )
        # normalise phase to [0, 2) in units of pi
        # take abs value of angle to account for phase wrapping (at 0, 2 pi)
        phases = np.abs(np.angle(results[np.ix_(focal_freqs, focal_freqs)]))
        phases[phases < 0] += 2 * np.pi
        phases /= np.pi
        assert np.isclose(np.nanmean(phases), phase, atol=close_atol)
