"""Tests for toolbox utilities."""

import pytest
import numpy as np
import scipy as sp
from mne import Info

from pybispectra.utils import (
    ResultsCFC,
    ResultsTDE,
    ResultsWaveShape,
    compute_fft,
    compute_tfr,
)
from pybispectra.utils._utils import (
    _compute_pearsonr_2d,
    _create_mne_info,
    _fast_find_first,
    _generate_data,
)


def test_results() -> None:
    """Test `ResultsCFC`."""
    n_cons = 9
    n_f1 = 50
    n_f2 = 50
    data = _generate_data(n_cons, n_f1, n_f2)

    f1s = np.arange(n_f1)
    f2s = np.arange(n_f2)

    n_unique_chans = 3
    indices = (
        np.repeat(np.arange(n_unique_chans), n_unique_chans).tolist(),
        np.tile(np.arange(n_unique_chans), n_unique_chans).tolist(),
    )

    # check if it runs with correct inputs
    results = ResultsCFC(
        data=data,
        indices=indices,
        f1s=f1s,
        f2s=f2s,
        name="test",
    )

    # check repr
    assert repr(results) == (
        f"'<Result: test | [{n_cons} nodes, {n_f1} f1s, {n_f2} f2s]>'"
    )

    # check if it catches incorrect inputs
    with pytest.raises(TypeError, match="`data` must be a NumPy array."):
        ResultsCFC(
            data=data.tolist(),
            indices=indices,
            f1s=f1s,
            f2s=f2s,
            name="test",
        )
    with pytest.raises(ValueError, match="`data` must be a 3D array."):
        ResultsCFC(
            data=data[:, :, 0],
            indices=indices,
            f1s=f1s,
            f2s=f2s,
            name="test",
        )

    with pytest.raises(TypeError, match="`indices` must be a tuple."):
        ResultsCFC(
            data=data,
            indices=list(indices),
            f1s=f1s,
            f2s=f2s,
            name="test",
        )
    with pytest.raises(ValueError, match="`indices` must have a length of 2."):
        ResultsCFC(
            data=data,
            indices=(indices[0], indices[0], indices[1]),
            f1s=f1s,
            f2s=f2s,
            name="test",
        )
    with pytest.raises(
        TypeError, match="Entries of `indices` must be NumPy arrays."
    ):
        ResultsCFC(
            data=data,
            indices=(indices[0][0], indices[1][0]),
            f1s=f1s,
            f2s=f2s,
            name="test",
        )
    with pytest.raises(
        ValueError, match="Entries of `indices` must be 1D arrays."
    ):
        ResultsCFC(
            data=data,
            indices=(
                np.vstack((indices[0], indices[0])),
                np.vstack((indices[1], indices[1])),
            ),
            f1s=f1s,
            f2s=f2s,
            name="test",
        )
    with pytest.raises(
        ValueError, match="Entries of `indices` must have the same length."
    ):
        ResultsCFC(
            data=data,
            indices=(indices[0], np.concatenate((indices[1], [1]))),
            f1s=f1s,
            f2s=f2s,
            name="test",
        )

    with pytest.raises(
        TypeError, match="`f1s` and `f2s` must be NumPy arrays."
    ):
        ResultsCFC(
            data=data,
            indices=indices,
            f1s=f1s.tolist(),
            f2s=f2s,
            name="test",
        )
    with pytest.raises(
        TypeError, match="`f1s` and `f2s` must be NumPy arrays."
    ):
        ResultsCFC(
            data=data,
            indices=indices,
            f1s=f1s,
            f2s=f2s.tolist(),
            name="test",
        )
    with pytest.raises(ValueError, match="`f1s` and `f2s` must be 1D arrays."):
        ResultsCFC(
            data=data,
            indices=indices,
            f1s=np.vstack((f1s, f1s)),
            f2s=f2s,
            name="test",
        )
    with pytest.raises(ValueError, match="`f1s` and `f2s` must be 1D arrays."):
        ResultsCFC(
            data=data,
            indices=indices,
            f1s=f1s,
            f2s=np.vstack((f2s, f2s)),
            name="test",
        )

    with pytest.raises(
        ValueError,
        match=r"`data` must have shape \[nodes, f1s, f2s\].",
    ):
        ResultsCFC(
            data=data[1:, :, :],
            indices=indices,
            f1s=f1s,
            f2s=f2s,
            name="test",
        )

    with pytest.raises(TypeError, match="`name` must be a string."):
        ResultsCFC(
            data=data,
            indices=indices,
            f1s=f1s,
            f2s=f2s,
            name=1,
        )

    # check if `get_results` works with correct inputs
    output = results.get_results(form="raveled")
    assert output.shape == (n_cons, n_f1, n_f2)
    output = results.get_results(form="compact")
    assert output[0].shape == (n_unique_chans, n_unique_chans, n_f1, n_f2)
    assert (idcs == np.arange(n_unique_chans) for idcs in output[1])

    # check if `get_results` catches incorrect inputs
    with pytest.raises(ValueError, match="`form` is not recognised."):
        results.get_results(form="notaform")

    # check if `plot` works with correct inputs
    figures, axes = results.plot(show=False)
    assert len(figures) == n_cons
    assert len(axes) == n_cons
    figures, axes = results.plot(n_rows=3, n_cols=3, show=True)
    assert len(figures) == 1
    assert len(axes) == 1

    # check if `plot` works with incorrect inputs
    with pytest.raises(TypeError, match="`nodes` must be a list of integers."):
        results.plot(nodes=9, show=False)
    with pytest.raises(TypeError, match="`nodes` must be a list of integers."):
        results.plot(nodes=[float(i) for i in range(n_cons)], show=False)
    with pytest.raises(
        ValueError, match="The requested connection is not present in the"
    ):
        results.plot(nodes=[-1], show=False)

    with pytest.raises(
        TypeError, match="`f1s` and `f2s` must be NumPy arrays."
    ):
        results.plot(f1s=0, show=False)
    with pytest.raises(
        TypeError, match="`f1s` and `f2s` must be NumPy arrays."
    ):
        results.plot(f2s=0, show=False)

    with pytest.raises(ValueError, match="`f1s` and `f2s` must be 1D arrays."):
        results.plot(f1s=np.random.rand(2, 2), show=False)
    with pytest.raises(ValueError, match="`f1s` and `f2s` must be 1D arrays."):
        results.plot(f2s=np.random.rand(2, 2), show=False)

    with pytest.raises(
        ValueError, match="Entries of `f1s` and `f2s` must be present in the"
    ):
        results.plot(f1s=f1s + 1, show=False)
    with pytest.raises(
        ValueError, match="Entries of `f1s` and `f2s` must be present in the"
    ):
        results.plot(f2s=f2s + 1, show=False)

    with pytest.raises(
        TypeError, match="`n_rows` and `n_cols` must be integers."
    ):
        results.plot(n_rows=0.5, show=False)
    with pytest.raises(
        TypeError, match="`n_rows` and `n_cols` must be integers."
    ):
        results.plot(n_cols=0.5, show=False)

    with pytest.raises(
        ValueError, match="`n_rows` and `n_cols` must be >= 1."
    ):
        results.plot(n_rows=0, show=False)
    with pytest.raises(
        ValueError, match="`n_rows` and `n_cols` must be >= 1."
    ):
        results.plot(n_cols=0, show=False)

    with pytest.raises(
        TypeError, match="`major_tick_intervals` and `minor_tick_intervals`"
    ):
        results.plot(major_tick_intervals="5", show=False)

    with pytest.raises(
        TypeError, match="`major_tick_intervals` and `minor_tick_intervals`"
    ):
        results.plot(minor_tick_intervals="1", show=False)


@pytest.mark.parametrize(
    "window", ["hanning", "hamming"], "return_neg_freqs", [True, False]
)
def test_compute_fft(window: str, return_neg_freqs: bool) -> None:
    """Test `compute_fft`."""
    n_epochs = 5
    n_chans = 3
    n_times = 100
    data = _generate_data(n_epochs, n_chans, n_times)
    sampling_freq = 50

    # check it runs with correct inputs
    fft, freqs = compute_fft(
        data=data,
        sampling_freq=sampling_freq,
        window=window,
        return_neg_freqs=return_neg_freqs,
        n_jobs=1,
    )
    assert isinstance(fft, np.ndarray), "`fft` should be a NumPy array."
    assert fft.ndim == 3, "`fft` should have 3 dimensions."
    assert fft.shape[:2] == (n_epochs, n_chans), (
        "The first 2 dimensions of `fft` should have shape [epochs x "
        "channels]."
    )
    assert isinstance(freqs, np.ndarray), "`freqs` should be a NumPy array."
    assert freqs.ndim == 1, "`freqs` should have 1 dimension."
    assert (
        fft.shape[2] == freqs.shape[0]
    ), "The 3rd dimension of `fft` should have the same length as `freqs`."
    assert (
        freqs[-1] == sampling_freq / 2
    ), "The maximum of `freqs` should be the Nyquist frequency."

    # NEED TO TEST FOR BEHAVIOUR WOTH RETURN_NEG_FREQS

    # check it catches incorrect inputs
    with pytest.raises(TypeError, match="`data` must be a NumPy array."):
        compute_fft(
            data=data.tolist(), sampling_freq=sampling_freq, window=window
        )
    with pytest.raises(ValueError, match="`data` must be a 3D array."):
        compute_fft(
            data=data[..., 0], sampling_freq=sampling_freq, window=window
        )
    with pytest.raises(ValueError, match="`data` must be real-valued."):
        compute_fft(
            data=np.array(data, dtype=np.complex128) + 1j,
            sampling_freq=sampling_freq,
        )

    with pytest.raises(
        TypeError, match="`sampling_freq` must be an int or a float."
    ):
        compute_fft(data=data, sampling_freq=[sampling_freq], window=window)

    with pytest.raises(TypeError, match="`n_points` must be an integer"):
        compute_fft(
            data=data, sampling_freq=sampling_freq, n_points=2.5, window=window
        )

    with pytest.raises(TypeError, match="`window` must be a str."):
        compute_fft(data=data, sampling_freq=sampling_freq, window=True)

    with pytest.raises(
        TypeError, match="The requested `window` type is not recognised."
    ):
        compute_fft(
            data=data, sampling_freq=sampling_freq, window="not_a_window"
        )

    with pytest.raises(TypeError, match="`return_neg_freqs` must be a bool."):
        compute_fft(
            data=data,
            sampling_freq=sampling_freq,
            window=window,
            return_neg_freqs="true",
        )

    with pytest.raises(TypeError, match="`n_jobs` must be an integer."):
        compute_fft(
            data=data, sampling_freq=sampling_freq, window=window, n_jobs=[]
        )
    with pytest.raises(ValueError, match="`n_jobs` must be >= 1."):
        compute_fft(
            data=data, sampling_freq=sampling_freq, window=window, n_jobs=0
        )

    with pytest.raises(TypeError, match="`verbose` must be a bool."):
        compute_fft(
            data=data,
            sampling_freq=sampling_freq,
            window=window,
            verbose="true",
        )

    # check it works with parallelisation (already tested without)
    compute_fft(
        data=data, sampling_freq=sampling_freq, window=window, n_jobs=2
    )
    compute_fft(
        data=data, sampling_freq=sampling_freq, window=window, n_jobs=-1
    )


@pytest.mark.parametrize("tfr_mode", ["morlet", "multitaper"])
def test_compute_tfr(tfr_mode: str) -> None:
    """Test `compute_tfr`."""
    n_epochs = 5
    n_chans = 3
    n_times = 100
    data = _generate_data(n_epochs, n_chans, n_times)
    sampling_freq = 100
    freqs_in = np.arange(20, 50)

    # check it runs with correct inputs
    tfr, freqs_out = compute_tfr(
        data=data,
        sampling_freq=sampling_freq,
        freqs=freqs_in,
        tfr_mode=tfr_mode,
        n_jobs=1,
    )
    assert isinstance(tfr, np.ndarray), "`tfr` should be a NumPy array."
    assert tfr.ndim == 4, "`tfr` should have 4 dimensions."
    assert tfr.shape[:3] == (n_epochs, n_chans, len(freqs_in)), (
        "The first 3 dimensions of `tfr` should have shape [epochs x "
        "channels x frequencies]."
    )
    assert isinstance(
        freqs_out, np.ndarray
    ), "`freqs_out` should be a NumPy array."
    assert np.all(
        freqs_in == freqs_out
    ), "`freqs_out` and `freqs_in` should be identical"

    # check it catches incorrect inputs
    with pytest.raises(TypeError, match="`data` must be a NumPy array."):
        compute_tfr(
            data=data.tolist(),
            sampling_freq=sampling_freq,
            freqs=freqs_in,
            tfr_mode=tfr_mode,
        )
    with pytest.raises(ValueError, match="`data` must be a 3D array."):
        compute_tfr(
            data=data[..., 0],
            sampling_freq=sampling_freq,
            freqs=freqs_in,
            tfr_mode=tfr_mode,
        )

    with pytest.raises(
        TypeError, match="`sampling_freq` must be an int or a float."
    ):
        compute_tfr(
            data=data,
            sampling_freq=[sampling_freq],
            freqs=freqs_in,
            tfr_mode=tfr_mode,
        )

    with pytest.raises(TypeError, match="`freqs` must be a NumPy array."):
        compute_tfr(
            data=data,
            sampling_freq=sampling_freq,
            freqs=freqs_in.tolist(),
            tfr_mode=tfr_mode,
        )
    with pytest.raises(ValueError, match="`freqs` must be a 1D array."):
        compute_tfr(
            data=data,
            sampling_freq=sampling_freq,
            freqs=freqs_in[:, np.newaxis],
            tfr_mode=tfr_mode,
        )
    with pytest.raises(
        ValueError, match="Entries of `freqs` must lie in the range"
    ):
        compute_tfr(
            data=data,
            sampling_freq=sampling_freq,
            freqs=np.arange(-1, sampling_freq * 0.5),
            tfr_mode=tfr_mode,
        )
    with pytest.raises(
        ValueError, match="Entries of `freqs` must lie in the range"
    ):
        compute_tfr(
            data=data,
            sampling_freq=sampling_freq,
            freqs=np.arange(0, sampling_freq),
            tfr_mode=tfr_mode,
        )
    with pytest.raises(
        ValueError, match="Entries of `freqs` must be in ascending order."
    ):
        compute_tfr(
            data=data,
            sampling_freq=sampling_freq,
            freqs=freqs_in[::-1],
            tfr_mode=tfr_mode,
        )

    with pytest.raises(TypeError, match="`tfr_mode` must be a str."):
        compute_tfr(
            data=data,
            sampling_freq=sampling_freq,
            freqs=freqs_in,
            tfr_mode=[tfr_mode],
        )
    with pytest.raises(ValueError, match="`tfr_mode` must be one of"):
        compute_tfr(
            data=data,
            sampling_freq=sampling_freq,
            freqs=freqs_in,
            tfr_mode="not_a_mode",
        )

    with pytest.raises(
        TypeError,
        match="`n_cycles` must be a NumPy array, an int, or a float.",
    ):
        compute_tfr(
            data=data,
            sampling_freq=sampling_freq,
            freqs=freqs_in,
            tfr_mode=tfr_mode,
            n_cycles=[3],
        )
    with pytest.raises(
        ValueError,
        match=(
            "If `n_cycles` is an array, it must have the same shape as "
            "`freqs`."
        ),
    ):
        compute_tfr(
            data=data,
            sampling_freq=sampling_freq,
            freqs=freqs_in,
            tfr_mode=tfr_mode,
            n_cycles=np.array([3]),
        )
    with pytest.raises(ValueError, match="`n_cycles` must be > 0."):
        compute_tfr(
            data=data,
            sampling_freq=sampling_freq,
            freqs=freqs_in,
            tfr_mode=tfr_mode,
            n_cycles=0,
        )
    with pytest.raises(ValueError, match="Entries of `n_cycles` must be > 0."):
        compute_tfr(
            data=data,
            sampling_freq=sampling_freq,
            freqs=freqs_in,
            tfr_mode=tfr_mode,
            n_cycles=np.zeros(len(freqs_in)),
        )

    with pytest.raises(
        TypeError, match="`zero_mean_wavelets` must be a bool or None."
    ):
        compute_tfr(
            data=data,
            sampling_freq=sampling_freq,
            freqs=freqs_in,
            tfr_mode=tfr_mode,
            zero_mean_wavelets="true",
        )

    with pytest.raises(TypeError, match="`use_fft` must be a bool."):
        compute_tfr(
            data=data,
            sampling_freq=sampling_freq,
            freqs=freqs_in,
            tfr_mode=tfr_mode,
            use_fft="true",
        )

    if tfr_mode == "multitaper":
        with pytest.raises(
            TypeError,
            match="`multitaper_time_bandwidth` must be an int or a float.",
        ):
            compute_tfr(
                data=data,
                sampling_freq=sampling_freq,
                freqs=freqs_in,
                tfr_mode=tfr_mode,
                multitaper_time_bandwidth=[3],
            )

    with pytest.raises(TypeError, match="`n_jobs` must be an integer."):
        compute_tfr(
            data=data,
            sampling_freq=sampling_freq,
            freqs=freqs_in,
            tfr_mode=tfr_mode,
            n_jobs=[],
        )
    with pytest.raises(ValueError, match="`n_jobs` must be >= 1."):
        compute_tfr(
            data=data,
            sampling_freq=sampling_freq,
            freqs=freqs_in,
            tfr_mode=tfr_mode,
            n_jobs=0,
        )

    with pytest.raises(TypeError, match="`verbose` must be a bool."):
        compute_tfr(
            data=data,
            sampling_freq=sampling_freq,
            freqs=freqs_in,
            tfr_mode=tfr_mode,
            verbose="true",
        )

    # check it works with parallelisation (already tested without)
    compute_tfr(
        data=data,
        sampling_freq=sampling_freq,
        freqs=freqs_in,
        tfr_mode=tfr_mode,
        n_jobs=2,
    )

    with pytest.warns(
        UserWarning, match="`data` is expected to be real-valued."
    ):
        compute_tfr(
            data=np.array(data, dtype=np.complex128) + 1j,
            sampling_freq=sampling_freq,
            freqs=freqs_in,
            tfr_mode=tfr_mode,
        )


def test_fast_find_first() -> None:
    """Test `_fast_find_first`."""
    vector = np.array([-1, 0, 1, 2, 3, 1])
    value = 1
    true_index = np.argwhere(vector == value)[0][0]

    # test that a present value is found
    index = _fast_find_first(vector=vector, value=value)
    assert (
        index == true_index
    ), f"The index of the value being found should be {true_index}."
    index = _fast_find_first(vector=vector, value=value, start_idx=1)
    assert (
        index == true_index
    ), f"The index of the value being found should be {true_index}."

    # test that a missing value is not found
    with pytest.raises(
        ValueError, match="`value` is not present in `vector`."
    ):
        _fast_find_first(vector=vector, value=4)
    with pytest.raises(
        ValueError, match="`value` is not present in `vector`."
    ):
        _fast_find_first(vector=vector, value=2, start_idx=4)


def test_compute_pearson_2d() -> None:
    """Test `_compute_pearsonr_2d`."""
    n_epochs = 5
    n_chans = 2
    n_times = 100
    data = _generate_data(n_epochs, n_chans, n_times)

    # test it works with correct inputs
    pearsonr = _compute_pearsonr_2d(x=data[:, 0], y=data[:, 1])
    assert isinstance(
        pearsonr, np.ndarray
    ), "`pearsonr` should be a NumPy array."
    assert pearsonr.ndim == 1, "`pearsonr` should be a 1D array."

    # test it matches the statistic output of SciPy's function
    sp_pearsonr = np.full(n_epochs, fill_value=np.nan)
    for epoch_i in range(n_epochs):
        sp_pearsonr[epoch_i] = sp.stats.pearsonr(
            data[epoch_i, 0], data[epoch_i, 1]
        )[0]

    assert np.allclose(
        pearsonr, sp_pearsonr
    ), "`pearsonr` should match the statistic of SciPy's function."


def test_create_mne_info() -> None:
    """Test `_create_mne_info`."""
    n_chans = 3
    sampling_freq = 50

    # test it works with correct inputs
    info = _create_mne_info(n_chans=n_chans, sampling_freq=sampling_freq)

    assert isinstance(info, Info), "`info` should be an MNE Info object."
    assert info["sfreq"] == sampling_freq, (
        "`info['sfreq']` should be equal" " to `sampling_freq`."
    )
    assert len(info["chs"]) == n_chans, (
        "`info['chs']` should have length" " `n_chans`."
    )
    assert info.get_channel_types() == [
        "eeg" for _ in range(n_chans)
    ], "Channels in `info` should be marked as EEG channels."
