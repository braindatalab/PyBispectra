"""Tests for toolbox utility functions."""

import os

import pytest
import numpy as np
import scipy as sp
from mne import Info

from pybispectra.data import get_example_data_paths, DATASETS
from pybispectra.utils import (
    compute_fft,
    compute_tfr,
    compute_rank,
    set_precision,
)
from pybispectra.utils._defaults import _precision, _Precision
from pybispectra.utils._utils import (
    _compute_pearsonr_2d,
    _create_mne_info,
    _fast_find_first,
    _generate_data,
)

set_precision("double")  # make sure precision is as default before testing


@pytest.mark.parametrize("window", ["hanning", "hamming"])
def test_compute_fft(window: str) -> None:
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
    assert freqs[0] == 0, "The first entry of `freqs` should be 0."
    max_freq_idx = np.where(freqs == freqs.max())[0][0]
    assert max_freq_idx != 0 and np.all(
        freqs[:max_freq_idx] == np.sort(freqs[:max_freq_idx])
    ), (
        "Entries of `freqs` corresponding to positive frequencies must be in "
        "ascending order."
    )
    assert (
        max(freqs) <= sampling_freq / 2
    ), "The maximum of `freqs` should be <= the Nyquist frequency."
    assert (
        freqs[-1] == sampling_freq / 2
    ), "The last entry of `freqs` should be the Nyquist frequency."

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
        ValueError, match="The requested `window` type is not recognised."
    ):
        compute_fft(
            data=data, sampling_freq=sampling_freq, window="not_a_window"
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
@pytest.mark.parametrize("zero_mean_wavelets", [True, False, None])
def test_compute_tfr(tfr_mode: str, zero_mean_wavelets: bool | None) -> None:
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
        zero_mean_wavelets=zero_mean_wavelets,
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
            zero_mean_wavelets=zero_mean_wavelets,
        )
    with pytest.raises(ValueError, match="`data` must be a 3D array."):
        compute_tfr(
            data=data[..., 0],
            sampling_freq=sampling_freq,
            freqs=freqs_in,
            tfr_mode=tfr_mode,
            zero_mean_wavelets=zero_mean_wavelets,
        )

    with pytest.raises(
        TypeError, match="`sampling_freq` must be an int or a float."
    ):
        compute_tfr(
            data=data,
            sampling_freq=[sampling_freq],
            freqs=freqs_in,
            tfr_mode=tfr_mode,
            zero_mean_wavelets=zero_mean_wavelets,
        )

    with pytest.raises(TypeError, match="`freqs` must be a NumPy array."):
        compute_tfr(
            data=data,
            sampling_freq=sampling_freq,
            freqs=freqs_in.tolist(),
            tfr_mode=tfr_mode,
            zero_mean_wavelets=zero_mean_wavelets,
        )
    with pytest.raises(ValueError, match="`freqs` must be a 1D array."):
        compute_tfr(
            data=data,
            sampling_freq=sampling_freq,
            freqs=freqs_in[:, np.newaxis],
            tfr_mode=tfr_mode,
            zero_mean_wavelets=zero_mean_wavelets,
        )
    with pytest.raises(
        ValueError, match="Entries of `freqs` must lie in the range"
    ):
        compute_tfr(
            data=data,
            sampling_freq=sampling_freq,
            freqs=np.arange(-1, sampling_freq * 0.5),
            tfr_mode=tfr_mode,
            zero_mean_wavelets=zero_mean_wavelets,
        )
    with pytest.raises(
        ValueError, match="Entries of `freqs` must lie in the range"
    ):
        compute_tfr(
            data=data,
            sampling_freq=sampling_freq,
            freqs=np.arange(0, sampling_freq),
            tfr_mode=tfr_mode,
            zero_mean_wavelets=zero_mean_wavelets,
        )
    with pytest.raises(
        ValueError, match="Entries of `freqs` must be in ascending order."
    ):
        compute_tfr(
            data=data,
            sampling_freq=sampling_freq,
            freqs=freqs_in[::-1],
            tfr_mode=tfr_mode,
            zero_mean_wavelets=zero_mean_wavelets,
        )

    with pytest.raises(TypeError, match="`tfr_mode` must be a str."):
        compute_tfr(
            data=data,
            sampling_freq=sampling_freq,
            freqs=freqs_in,
            tfr_mode=[tfr_mode],
            zero_mean_wavelets=zero_mean_wavelets,
        )
    with pytest.raises(ValueError, match="`tfr_mode` must be one of"):
        compute_tfr(
            data=data,
            sampling_freq=sampling_freq,
            freqs=freqs_in,
            tfr_mode="not_a_mode",
            zero_mean_wavelets=zero_mean_wavelets,
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
            zero_mean_wavelets=zero_mean_wavelets,
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
            zero_mean_wavelets=zero_mean_wavelets,
        )
    with pytest.raises(ValueError, match="`n_cycles` must be > 0."):
        compute_tfr(
            data=data,
            sampling_freq=sampling_freq,
            freqs=freqs_in,
            tfr_mode=tfr_mode,
            n_cycles=0,
            zero_mean_wavelets=zero_mean_wavelets,
        )
    with pytest.raises(ValueError, match="Entries of `n_cycles` must be > 0."):
        compute_tfr(
            data=data,
            sampling_freq=sampling_freq,
            freqs=freqs_in,
            tfr_mode=tfr_mode,
            n_cycles=np.zeros(len(freqs_in)),
            zero_mean_wavelets=zero_mean_wavelets,
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
            zero_mean_wavelets=zero_mean_wavelets,
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
                zero_mean_wavelets=zero_mean_wavelets,
                multitaper_time_bandwidth=[3],
            )

    with pytest.raises(TypeError, match="`n_jobs` must be an integer."):
        compute_tfr(
            data=data,
            sampling_freq=sampling_freq,
            freqs=freqs_in,
            tfr_mode=tfr_mode,
            zero_mean_wavelets=zero_mean_wavelets,
            n_jobs=[],
        )
    with pytest.raises(ValueError, match="`n_jobs` must be >= 1."):
        compute_tfr(
            data=data,
            sampling_freq=sampling_freq,
            freqs=freqs_in,
            tfr_mode=tfr_mode,
            zero_mean_wavelets=zero_mean_wavelets,
            n_jobs=0,
        )

    with pytest.raises(TypeError, match="`verbose` must be a bool."):
        compute_tfr(
            data=data,
            sampling_freq=sampling_freq,
            freqs=freqs_in,
            tfr_mode=tfr_mode,
            zero_mean_wavelets=zero_mean_wavelets,
            verbose="true",
        )

    # check it works with parallelisation (already tested without)
    compute_tfr(
        data=data,
        sampling_freq=sampling_freq,
        freqs=freqs_in,
        tfr_mode=tfr_mode,
        zero_mean_wavelets=zero_mean_wavelets,
        n_jobs=2,
    )
    compute_tfr(
        data=data,
        sampling_freq=sampling_freq,
        freqs=freqs_in,
        tfr_mode=tfr_mode,
        zero_mean_wavelets=zero_mean_wavelets,
        n_jobs=-1,
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


def test_compute_rank() -> None:
    """Test `compute_rank`."""
    n_epochs = 5
    n_chans = 3
    n_times = 100
    data = _generate_data(n_epochs, n_chans, n_times)

    # test it works with correct inputs
    rank = compute_rank(data=data)
    assert isinstance(rank, int), "`rank` should be an int."
    non_full_rank_data = data.copy()
    non_full_rank_data[:, 1] = non_full_rank_data[:, 0]
    rank = compute_rank(data=non_full_rank_data)
    assert rank == n_chans - 1, "`rank` should be equal to n_chans - 1."

    # test it catches incorrect inputs
    with pytest.raises(TypeError, match="`data` must be a NumPy array."):
        compute_rank(data=data.tolist())
    with pytest.raises(ValueError, match="`data` must be a 3D array."):
        compute_rank(data=data[0])

    with pytest.raises(TypeError, match="`sv_tol` must be a float or an int."):
        compute_rank(data=data, sv_tol=None)


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
    pearsonr = _compute_pearsonr_2d(
        x=data[:, 0], y=data[:, 1], precision=_precision.real
    )
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


def test_get_example_data_paths() -> None:
    """Test `get_example_data_paths`."""
    # test it works with correct inputs
    for name, file in DATASETS.items():
        path = get_example_data_paths(name=name)
        assert isinstance(path, str), "`path` should be a str."
        assert path.endswith(
            file
        ), "`path` should end with the name of the dataset."
        assert os.path.exists(path), "`path` should point to an existing file."

    # test it catches incorrect inputs
    with pytest.raises(ValueError, match="`name` must be one of"):
        get_example_data_paths(name="not_a_name")


@pytest.mark.parametrize("precision_object", [_precision, _precision])
@pytest.mark.parametrize("precision_type", ["single", "double"])
def test_set_precision(
    precision_object: _Precision, precision_type: str
) -> None:
    """Test `set_precision`."""
    # error catching
    with pytest.raises(
        ValueError, match="precision must be either 'single' or 'double'."
    ):
        set_precision(precision="not_a_precision")

    # default precision should be double
    assert precision_object.type == "double"
    assert precision_object.real == np.float64
    assert precision_object.complex == np.complex128

    set_precision(precision=precision_type)

    if precision_type == "single":
        assert precision_object.type == "single"
        assert precision_object.real == np.float32
        assert precision_object.complex == np.complex64
        # reset precision to default
        set_precision("double")
    else:
        assert precision_object.type == "double"
        assert precision_object.real == np.float64
        assert precision_object.complex == np.complex128
