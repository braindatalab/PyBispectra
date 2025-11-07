"""Tests for CFC tools."""

import numpy as np
import pytest

from pybispectra.cfc import AAC, PAC, PPC
from pybispectra.utils import (
    ResultsCFC,
    compute_fft,
    compute_tfr,
    get_example_data_paths,
)
from pybispectra.utils._utils import _generate_data


@pytest.mark.parametrize("class_type", ["PAC", "PPC", "AAC"])
def test_error_catch(class_type: str) -> None:
    """Check that CFC classes catch errors."""
    n_chans = 3
    n_epochs = 5
    n_times = 100
    sampling_freq = 50
    data = _generate_data((n_epochs, n_chans, n_times), complexobj=False)
    indices = ([0, 1, 2], [0, 1, 2])
    freqs = np.arange(5, 20)

    if class_type == "PAC":
        TestClass = PAC
        coeffs, freqs = compute_fft(data, sampling_freq)
    elif class_type == "PPC":
        TestClass = PPC
        coeffs, freqs = compute_tfr(
            data, sampling_freq, freqs, n_cycles=3, output="complex"
        )
    else:
        TestClass = AAC
        coeffs, freqs = compute_tfr(
            data, sampling_freq, freqs, n_cycles=3, output="power"
        )

    # initialisation
    with pytest.raises(TypeError, match="`data` must be a NumPy array."):
        TestClass(coeffs.tolist(), freqs, sampling_freq)
    if class_type == "PAC":
        with pytest.raises(ValueError, match="`data` must be a 3D or 4D array."):
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
        match=("`data` and `freqs` must contain the same number of frequencies."),
    ):
        TestClass(coeffs, freqs[:-1], sampling_freq)

    with pytest.raises(ValueError, match="Entries of `freqs` must be >= 0."):
        TestClass(coeffs, freqs * -1, sampling_freq)
    with pytest.raises(
        ValueError,
        match="At least one entry of `freqs` is > the Nyquist frequency.",
    ):
        bad_freqs = np.linspace(0, sampling_freq / 2 + 1, freqs.size)
        TestClass(coeffs, bad_freqs, sampling_freq)
    with pytest.raises(
        ValueError,
        match=("Entries of `freqs` must be in ascending order."),
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

    # test that errors for incorrect inputs are caught
    if class_type == "PAC":
        with pytest.raises(
            TypeError,
            match="`antisym` must be a bool or tuple of bools.",
        ):
            test_class.compute(antisym="true")
        with pytest.raises(TypeError, match="Entries of `antisym` must be bools."):
            test_class.compute(antisym=("true",))

        with pytest.raises(TypeError, match="`norm` must be a bool or tuple of bools."):
            test_class.compute(norm="true")
        with pytest.raises(TypeError, match="Entries of `norm` must be bools."):
            test_class.compute(norm=("true",))

    with pytest.raises(TypeError, match="`indices` must be a tuple."):
        test_class.compute(indices=list(indices))
    with pytest.raises(ValueError, match="`indices` must have length of 2."):
        test_class.compute(indices=(0, 1, 2))
    with pytest.raises(TypeError, match="Entries of `indices` must be tuples."):
        test_class.compute(indices=(0, 1))
    with pytest.raises(
        TypeError, match="Entries for seeds and targets in `indices` must be ints."
    ):
        test_class.compute(indices=((0.0,), (1.0,)))
    with pytest.raises(
        ValueError,
        match=("`indices` contains indices for channels not present in the data."),
    ):
        test_class.compute(indices=((0,), (99,)))
    with pytest.raises(
        ValueError, match="Entries of `indices` must have equal length."
    ):
        test_class.compute(indices=((0,), (1, 2)))

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


@pytest.mark.parametrize("class_type", ["PAC", "PPC", "AAC"])
def test_error_catch_time_resolved(class_type: str) -> None:
    """Check that CFC classes catch errors for time-resolved data."""
    n_chans = 3
    n_epochs = 5
    n_times = 100
    sampling_freq = 50
    data = _generate_data((n_epochs, n_chans, n_times), complexobj=False)
    freqs = np.arange(5, 20)
    times = np.arange(n_times) / sampling_freq

    output = "power" if class_type == "AAC" else "complex"
    coeffs, freqs = compute_tfr(data, sampling_freq, freqs, n_cycles=3, output=output)

    if class_type == "PAC":
        TestClass = PAC
    elif class_type == "PPC":
        TestClass = PPC
    else:
        TestClass = AAC

    # initialisation
    if class_type == "PAC":
        with pytest.raises(ValueError, match="`data` must be a 3D or 4D array."):
            TestClass(np.random.randn(2, 2), freqs, sampling_freq)
    else:
        with pytest.raises(ValueError, match="`data` must be a 4D array."):
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


def test_pac_runs() -> None:
    """Test that PAC runs correctly."""
    n_chans = 3
    n_times = 100
    sampling_freq = 50
    data = _generate_data((5, n_chans, n_times), complexobj=False)
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
    pac = PAC(data=fft, freqs=freqs, sampling_freq=sampling_freq)
    assert np.all(pac.data == fft), "FFT data not stored correctly"
    pac_tr = PAC(data=tfr, freqs=freqs, sampling_freq=sampling_freq)
    assert np.all(pac_tr.data == tfr), "TFR data not stored correctly"

    # check times are handled correctly
    pac = PAC(data=fft, freqs=freqs, sampling_freq=sampling_freq, times=times)
    assert pac.times is None, "`times` should be ignored for time-resolved_data"
    pac = PAC(data=tfr, freqs=freqs, sampling_freq=sampling_freq, times=times)
    assert np.all(pac.times == times), "`times` should be stored for time-resolved_data"
    pac = PAC(data=tfr, freqs=freqs, sampling_freq=sampling_freq)
    assert np.all(pac.times == default_times), (
        "Auto-generated `times` are incorrect for time-resolved_data"
    )

    # check it runs with correct inputs
    pac = PAC(data=fft, freqs=freqs, sampling_freq=sampling_freq)
    pac.compute(antisym=(False, True), norm=(False, True))
    pac_tr = PAC(data=tfr, freqs=freqs, sampling_freq=sampling_freq, times=times)
    pac_tr.compute()

    # check the returned results have the correct shape
    assert (
        results.shape == (n_chans**2, len(freqs), len(freqs)) for results in pac.results
    )
    assert pac_tr.results.shape == (n_chans**2, len(freqs), len(freqs), len(times))

    # check the returned results are of the correct type
    result_types = [
        "PAC | Bispectrum",
        "PAC | Bicoherence",
        "PAC (antisymmetrised) | Bispectrum",
        "PAC (antisymmetrised) | Bicoherence",
    ]
    assert (results.name == result_types[i] for i, results in enumerate(pac.results))
    assert (isinstance(results, ResultsCFC) for results in pac.results)

    pac.compute(antisym=False, norm=False)
    assert isinstance(pac.results, ResultsCFC)
    assert pac.results.name == result_types[0]

    pac.compute(antisym=False, norm=True)
    assert isinstance(pac.results, ResultsCFC)
    assert pac.results.name == result_types[1]

    pac.compute(antisym=True, norm=False)
    assert isinstance(pac.results, ResultsCFC)
    assert pac.results.name == result_types[2]

    pac.compute(antisym=True, norm=True)
    assert isinstance(pac.results, ResultsCFC)
    assert pac.results.name == result_types[3]

    pac.compute(antisym=(False, True), norm=False)
    assert len(pac.results) == 2
    assert (
        pac.results[i].name == result_types[type_i] for i, type_i in enumerate((0, 2))
    )

    pac.compute(antisym=False, norm=(False, True))
    assert len(pac.results) == 2
    assert (
        pac.results[i].name == result_types[type_i] for i, type_i in enumerate((0, 3))
    )

    # check it runs with non-exact frequencies
    fmin, fmax = 10.25, 19.75
    freqs_sel = freqs[np.argwhere((freqs >= fmin) & (freqs <= fmax)).squeeze()]
    pac.compute(f1s=(fmin, fmax), f2s=(fmin, fmax))
    assert pac.results.get_results().shape[1:3] == (len(freqs_sel), len(freqs_sel)), (
        "Number of frequencies in results does not match the selection"
    )
    assert (
        pac.results.f1s[0] == freqs_sel[0]
        and pac.results.f1s[-1] == freqs_sel[-1]
        and pac.results.f2s[0] == freqs_sel[0]
        and pac.results.f2s[-1] == freqs_sel[-1]
    ), "`f1s` and `f2s` in results do not match the selection"

    # check it runs with non-exact times
    tmin, tmax = 10.55, 11.55
    times_sel = times[np.argwhere((times >= tmin) & (times <= tmax)).squeeze()]
    pac_tr.compute(times=(tmin, tmax))
    assert pac_tr.results.get_results().shape[3] == len(times_sel), (
        "Number of timepoints in results does not match the selection"
    )
    assert (
        pac_tr.results.times[0] == times_sel[0]
        and pac_tr.results.times[-1] == times_sel[-1]
    ), "`times` in results do not match the selection"

    # test that time selection works
    tmin_idx, tmax_idx = 5, 10
    pac_tr.compute(
        antisym=(False, True),
        norm=(False, True),
        times=(times[tmin_idx], times[tmax_idx]),
    )
    pac_tr_results = pac_tr.results
    pac_tr_window = PAC(
        data=tfr[..., tmin_idx : tmax_idx + 1],
        freqs=freqs,
        sampling_freq=sampling_freq,
        times=times[tmin_idx : tmax_idx + 1],
    )
    pac_tr_window.compute(antisym=(False, True), norm=(False, True))
    pac_tr_window_results = pac_tr_window.results
    for results, window_results in zip(pac_tr_results, pac_tr_window_results):
        assert np.all(results.times == window_results.times)
        assert np.array_equal(
            results.get_results(), window_results.get_results(), equal_nan=True
        )

    # test it runs with parallelisation
    pac.compute(n_jobs=2)
    pac.compute(n_jobs=-1)

    # test copying works
    pac_copy = pac.copy()
    attrs = pac.__dict__.keys()
    for attr in attrs:
        if not attr.startswith("_"):
            assert np.all(getattr(pac, attr) == getattr(pac_copy, attr))
    assert pac is not pac_copy


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
        data=data, sampling_freq=sampling_freq, n_points=sampling_freq
    )

    # compute PAC
    pac = PAC(data=fft_coeffs, freqs=freqs, sampling_freq=sampling_freq)
    pac.compute(indices=((0,), (1,)), antisym=False, norm=False)
    results = pac.results.get_results()

    # check that 10-60 Hz PAC is detected
    assert (
        results[0][np.ix_(interacting_f1s, interacting_f2s)].mean()
        > results[0][np.ix_(noninteracting_f1s, noninteracting_f2s)].mean()
    )

    # Test that spurious PAC is corrected with antisymmetrisation
    # load simulated data with univariate PAC interactions
    data = np.load(get_example_data_paths("sim_data_pac_univariate"))
    sampling_freq = 200  # sampling frequency in Hz

    # compute FFT
    fft_coeffs, freqs = compute_fft(
        data=data,
        sampling_freq=sampling_freq,
        n_points=sampling_freq,
    )

    # compute PAC
    pac = PAC(data=fft_coeffs, freqs=freqs, sampling_freq=sampling_freq)
    pac.compute(indices=((0, 1, 0), (0, 1, 1)), antisym=False, norm=False)
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
    pac.compute(indices=((0,), (1,)), antisym=True, norm=False)
    results = pac.results.get_results()
    assert np.isclose(
        results[0][np.ix_(interacting_f1s, interacting_f2s)].mean(),
        results[0][np.ix_(noninteracting_f1s, noninteracting_f2s)].mean(),
        atol=1e-4,
    )


def test_ppc_runs() -> None:
    """Test that PPC runs correctly."""
    n_chans = 3
    n_times = 100
    sampling_freq = 50
    data = _generate_data((5, n_chans, n_times), complexobj=False)
    default_times = np.arange(n_times) / sampling_freq  # matches auto-generated times
    times = default_times + 10  # offset to distinguish from auto-generated ones
    freqs = np.arange(5, 20)

    tfr, freqs = compute_tfr(
        data=data,
        sampling_freq=sampling_freq,
        freqs=freqs,
        n_cycles=3,
        output="complex",
        verbose=False,
    )

    # check it runs with correct inputs
    ppc = PPC(data=tfr, freqs=freqs, sampling_freq=sampling_freq, times=times)
    ppc.compute()

    # check the returned results have the correct shape
    assert ppc.results.shape == (n_chans**2, len(freqs), len(freqs))

    # check the returned results are of the correct type
    assert ppc.results.name == "PPC"
    assert isinstance(ppc.results, ResultsCFC)

    # check it runs with non-exact frequencies
    fmin, fmax = 10.25, 19.75
    freqs_sel = freqs[np.argwhere((freqs >= fmin) & (freqs <= fmax)).squeeze()]
    ppc.compute(f1s=(fmin, fmax), f2s=(fmin, fmax))
    assert ppc.results.get_results().shape[1:3] == (len(freqs_sel), len(freqs_sel)), (
        "Number of frequencies in results does not match the selection"
    )
    assert (
        ppc.results.f1s[0] == freqs_sel[0]
        and ppc.results.f1s[-1] == freqs_sel[-1]
        and ppc.results.f2s[0] == freqs_sel[0]
        and ppc.results.f2s[-1] == freqs_sel[-1]
    ), "`f1s` and `f2s` in results do not match the selection"

    # check it runs with non-exact times
    ppc.compute(times=(10.55, 11.55))

    # test that time selection works
    tmin_idx, tmax_idx = 5, 10
    ppc.compute(times=(times[tmin_idx], times[tmax_idx]))
    ppc_results = ppc.results
    ppc_window = PPC(
        data=tfr[..., tmin_idx : tmax_idx + 1],
        freqs=freqs,
        sampling_freq=sampling_freq,
        times=times[tmin_idx : tmax_idx + 1],
    )
    ppc_window.compute()
    ppc_window_results = ppc_window.results
    assert np.array_equal(
        ppc_results.get_results(), ppc_window_results.get_results(), equal_nan=True
    )

    # test it runs with parallelisation
    ppc.compute(n_jobs=2)
    ppc.compute(n_jobs=-1)

    # test copying works
    ppc_copy = ppc.copy()
    attrs = ppc.__dict__.keys()
    for attr in attrs:
        if not attr.startswith("_"):
            assert np.all(getattr(ppc, attr) == getattr(ppc_copy, attr))
    assert ppc is not ppc_copy


def test_ppc_results():
    """Test that PPC returns the correct results.

    Simulated data with 10-60 Hz PPC between channels 0 and 1 is used.
    """
    # identify interacting and non-interacting frequencies (10-60 Hz PPC)
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

    # test that genuine PPC is detected
    data = np.load(get_example_data_paths("sim_data_ppc"))
    sampling_freq = 500  # sampling frequency in Hz
    freqs = np.arange(1, 100)

    # compute TFR
    tfr_coeffs, freqs = compute_tfr(
        data=data,
        sampling_freq=sampling_freq,
        freqs=freqs,
        n_cycles=freqs / 1.25,
        output="complex",
    )

    # compute PPC
    ppc = PPC(data=tfr_coeffs, freqs=freqs, sampling_freq=sampling_freq)
    ppc.compute(indices=((0, 0, 1), (0, 1, 1)))
    results = ppc.results.get_results()

    # check that 10-60 Hz PPC is detected for connection 0 → 1 at interacting freqs
    assert (
        results[1][np.ix_(interacting_f1s, interacting_f2s)].mean()
        > results[1][np.ix_(noninteracting_f1s, noninteracting_f2s)].mean()
    ), "0 → 1 PPC at interacting freqs not greater than at non-interacting freqs"
    assert (
        results[1][np.ix_(interacting_f1s, interacting_f2s)].mean()
        > results[0][np.ix_(interacting_f1s, interacting_f2s)].mean()
    ), "0 → 1 PPC at interacting freqs not greater than for 0 → 0"
    assert (
        results[1][np.ix_(interacting_f1s, interacting_f2s)].mean()
        > results[2][np.ix_(interacting_f1s, interacting_f2s)].mean()
    ), "0 → 1 PPC at interacting freqs not greater than for 1 → 1"


def test_aac_runs() -> None:
    """Test that AAC runs correctly."""
    n_chans = 3
    n_times = 100
    sampling_freq = 50
    data = _generate_data((5, n_chans, n_times), complexobj=False)
    default_times = np.arange(n_times) / sampling_freq  # matches auto-generated times
    times = default_times + 10  # offset to distinguish from auto-generated ones
    freqs = np.arange(5, 25, 0.5)

    tfr, freqs = compute_tfr(data, sampling_freq, freqs, n_cycles=3)

    # check data is stored correctly
    aac = AAC(data=tfr, freqs=freqs, sampling_freq=sampling_freq)
    assert np.all(aac.data == tfr), "TFR data not stored correctly"

    # check times are handled correctly
    aac = AAC(data=tfr, freqs=freqs, sampling_freq=sampling_freq, times=times)
    assert np.all(aac.times == times), "`times` should be stored for time-resolved_data"
    aac = AAC(data=tfr, freqs=freqs, sampling_freq=sampling_freq)
    assert np.all(aac.times == default_times), (
        "Auto-generated `times` are incorrect for time-resolved_data"
    )

    # check it runs with correct inputs
    aac = AAC(data=tfr, freqs=freqs, sampling_freq=sampling_freq, times=times)
    aac.compute()

    # check the returned results have the correct shape
    assert aac.results.shape == (n_chans**2, len(freqs), len(freqs))

    # check the returned results are of the correct type
    assert aac.results.name == "AAC"
    assert isinstance(aac.results, ResultsCFC)

    # check it runs with non-exact frequencies
    fmin, fmax = 10.25, 19.75
    freqs_sel = freqs[np.argwhere((freqs >= fmin) & (freqs <= fmax)).squeeze()]
    aac.compute(f1s=(fmin, fmax), f2s=(fmin, fmax))
    assert aac.results.get_results().shape[1:3] == (len(freqs_sel), len(freqs_sel)), (
        "Number of frequencies in results does not match the selection"
    )
    assert (
        aac.results.f1s[0] == freqs_sel[0]
        and aac.results.f1s[-1] == freqs_sel[-1]
        and aac.results.f2s[0] == freqs_sel[0]
        and aac.results.f2s[-1] == freqs_sel[-1]
    ), "`f1s` and `f2s` in results do not match the selection"

    # check it runs with non-exact times
    aac.compute(times=(10.55, 11.55))

    # test that time selection works
    tmin_idx, tmax_idx = 5, 10
    aac.compute(times=(times[tmin_idx], times[tmax_idx]))
    aac_results = aac.results
    aac_window = AAC(
        data=tfr[..., tmin_idx : tmax_idx + 1],
        freqs=freqs,
        sampling_freq=sampling_freq,
        times=times[tmin_idx : tmax_idx + 1],
    )
    aac_window.compute()
    aac_window_results = aac_window.results
    assert np.array_equal(
        aac_results.get_results(), aac_window_results.get_results(), equal_nan=True
    )

    # test it runs with parallelisation
    aac.compute(n_jobs=2)
    aac.compute(n_jobs=-1)

    # test copying works
    aac_copy = aac.copy()
    attrs = aac.__dict__.keys()
    for attr in attrs:
        if not attr.startswith("_"):
            assert np.all(getattr(aac, attr) == getattr(aac_copy, attr))
    assert aac is not aac_copy
