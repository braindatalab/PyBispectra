"""Tests for generalised eigendecomposition tools."""

import numpy as np
import pytest

from pybispectra.utils import SpatioSpectralFilter
from pybispectra.utils._utils import _generate_data
from pybispectra.utils._defaults import _precision


@pytest.mark.parametrize("method", ["ssd", "hpmax"])
def test_error_catch(method: str) -> None:
    """Check that SpatioSpectralFilter class catches errors."""
    n_chans = 3
    n_epochs = 5
    n_times = 200
    sampling_freq = 50
    data = _generate_data((n_epochs, n_chans, n_times), complexobj=False)
    signal_bounds = (10, 15)
    noise_bounds = (8, 17)

    # initialisation
    with pytest.raises(TypeError, match="`data` must be a NumPy array."):
        SpatioSpectralFilter(data.tolist(), sampling_freq)
    with pytest.raises(ValueError, match="`data` must be a 3D array."):
        SpatioSpectralFilter(data[0], sampling_freq)
    with pytest.raises(TypeError, match="`data` must be a real-valued object."):
        SpatioSpectralFilter(data.astype(_precision.complex), sampling_freq)

    with pytest.raises(TypeError, match="`sampling_freq` must be an int or a float."):
        SpatioSpectralFilter(data, [sampling_freq])

    ssf = SpatioSpectralFilter(data, sampling_freq)

    # SSD and HPMax inputs
    with pytest.raises(
        TypeError, match="`signal_bounds` must be a tuple of ints or floats."
    ):
        ssf.fit_transform_ssd(signal_bounds=[10, 15], noise_bounds=noise_bounds)
    with pytest.raises(
        TypeError, match="`signal_bounds` must be a tuple of ints or floats."
    ):
        ssf.fit_transform_ssd(signal_bounds=(None, None), noise_bounds=noise_bounds)
    with pytest.raises(
        ValueError, match="`signal_bounds` and `noise_bounds` must have lengths of 2."
    ):
        ssf.fit_transform_ssd(signal_bounds=(10, 15, 30), noise_bounds=noise_bounds)

    with pytest.raises(
        TypeError, match="`noise_bounds` must be a tuple of ints or floats."
    ):
        ssf.fit_transform_ssd(signal_bounds=signal_bounds, noise_bounds=[8, 17])
    with pytest.raises(
        TypeError, match="`noise_bounds` must be a tuple of ints or floats."
    ):
        ssf.fit_transform_ssd(signal_bounds=signal_bounds, noise_bounds=(None, None))
    with pytest.raises(
        ValueError, match="`signal_bounds` and `noise_bounds` must have lengths of 2."
    ):
        ssf.fit_transform_ssd(signal_bounds=signal_bounds, noise_bounds=(8, 17, 30))

    with pytest.raises(TypeError, match="`rank` must be an int."):
        ssf.fit_transform_ssd(
            signal_bounds=signal_bounds, noise_bounds=noise_bounds, rank=1.0
        )
    with pytest.raises(
        ValueError,
        match="`rank` must be >= 1 and <= the number of channels in `indices`",
    ):
        ssf.fit_transform_ssd(
            signal_bounds=signal_bounds, noise_bounds=noise_bounds, rank=0
        )
    with pytest.raises(
        ValueError,
        match="`rank` must be >= 1 and <= the number of channels in `indices`",
    ):
        ssf.fit_transform_ssd(
            signal_bounds=signal_bounds, noise_bounds=noise_bounds, rank=n_chans + 1
        )
    with pytest.raises(
        ValueError,
        match="`rank` must be >= 1 and <= the number of channels in `indices`",
    ):
        ssf.fit_transform_ssd(
            signal_bounds=signal_bounds, noise_bounds=noise_bounds, indices=(0,), rank=2
        )

    with pytest.raises(TypeError, match="`indices` must be a tuple of ints."):
        ssf.fit_transform_ssd(
            signal_bounds=signal_bounds, noise_bounds=noise_bounds, indices=0
        )
    with pytest.raises(TypeError, match="`indices` must be a tuple of ints."):
        ssf.fit_transform_ssd(
            signal_bounds=signal_bounds, noise_bounds=noise_bounds, indices=(0.0,)
        )
    with pytest.raises(
        ValueError,
        match=(
            "`indices` can only contain channel indices >= 0 or < the number of "
            "channels in the data."
        ),
    ):
        ssf.fit_transform_ssd(
            signal_bounds=signal_bounds, noise_bounds=noise_bounds, indices=(-1,)
        )
    with pytest.raises(
        ValueError,
        match=(
            "`indices` can only contain channel indices >= 0 or < the number of "
            "channels in the data."
        ),
    ):
        ssf.fit_transform_ssd(
            signal_bounds=signal_bounds,
            noise_bounds=noise_bounds,
            indices=(n_chans + 1,),
        )

    # SSD inputs only
    if method == "ssd":
        with pytest.raises(
            TypeError, match="`signal_noise_gap` must be an int or a float."
        ):
            ssf.fit_transform_ssd(
                signal_bounds=signal_bounds,
                noise_bounds=noise_bounds,
                signal_noise_gap=None,
            )

        with pytest.raises(
            ValueError,
            match=(
                "The frequencies of `noise_bounds` must lie outside of `signal_bounds`"
            ),
        ):
            ssf.fit_transform_ssd(
                signal_bounds=(10, 15),
                noise_bounds=(10, 16),
                signal_noise_gap=1,
            )
        with pytest.raises(
            ValueError,
            match=(
                "The frequencies of `noise_bounds` must lie outside of `signal_bounds`"
            ),
        ):
            ssf.fit_transform_ssd(
                signal_bounds=(10, 15), noise_bounds=(9, 15), signal_noise_gap=1
            )

        with pytest.raises(TypeError, match="`bandpass_filter` must be a bool."):
            ssf.fit_transform_ssd(
                signal_bounds=signal_bounds,
                noise_bounds=noise_bounds,
                bandpass_filter="true",
            )

    # HPMax inputs only
    elif method == "hpmax":
        with pytest.raises(TypeError, match="`n_harmonics` must be an int."):
            ssf.fit_transform_hpmax(
                signal_bounds=signal_bounds, noise_bounds=noise_bounds, n_harmonics=0.5
            )

        with pytest.raises(ValueError, match="`n_harmonics` must be >= -1."):
            ssf.fit_transform_hpmax(
                signal_bounds=signal_bounds, noise_bounds=noise_bounds, n_harmonics=-2
            )

        with pytest.raises(
            ValueError,
            match=(
                "`n_harmonics` for the requested signal and noise frequencies extends "
                "beyond the Nyquist frequency."
            ),
        ):
            ssf.fit_transform_hpmax(
                signal_bounds=signal_bounds,
                noise_bounds=noise_bounds,
                n_harmonics=int(
                    np.ceil(((sampling_freq / 2) + noise_bounds[1]) / signal_bounds[1])
                ),
            )

        with pytest.raises(ValueError, match="`csd_method` is not recognised."):
            ssf.fit_transform_hpmax(
                signal_bounds=signal_bounds,
                noise_bounds=noise_bounds,
                csd_method="not_a_method",
            )

        with pytest.raises(TypeError, match="`n_jobs` must be an integer."):
            ssf.fit_transform_hpmax(
                signal_bounds=signal_bounds, noise_bounds=noise_bounds, n_jobs=0.5
            )
        with pytest.raises(ValueError, match="`n_jobs` must be >= 1 or -1."):
            ssf.fit_transform_hpmax(
                signal_bounds=signal_bounds, noise_bounds=noise_bounds, n_jobs=-2
            )

    # transform data
    with pytest.raises(ValueError, match="No filters have been fit."):
        ssf.transform()

    # get transformed data
    with pytest.raises(ValueError, match="No data has been transformed."):
        ssf.get_transformed_data()

    fit_method = ssf.fit_ssd if method == "ssd" else ssf.fit_hpmax
    fit_method(signal_bounds=signal_bounds, noise_bounds=noise_bounds)

    with pytest.raises(TypeError, match="`data` must be a NumPy array."):
        ssf.transform(data=data.tolist())
    with pytest.raises(ValueError, match="`data` must be a 3D array."):
        ssf.transform(data=data[0])
    with pytest.raises(
        ValueError, match="`data` must have the same number of channels as the filters."
    ):
        ssf.transform(data=data[:, 1:])

    ssf.transform()

    # return transformed data
    with pytest.raises(TypeError, match="`min_ratio` must be an int or a float"):
        ssf.get_transformed_data(min_ratio=None)
    with pytest.warns(
        UserWarning,
        match=(
            "No signal-to-noise ratios are greater than the requested minimum; "
            "returning an empty array."
        ),
    ):
        ssf.get_transformed_data(min_ratio=ssf.ratios.max() + 1)
    with pytest.raises(TypeError, match="`copy` must be a bool."):
        ssf.get_transformed_data(copy="True")


@pytest.mark.filterwarnings(
    r"ignore:No signal\-to\-noise ratios are greater than the requested minimum:"
    "UserWarning"
)
@pytest.mark.parametrize("bandpass_filter", [True, False])
@pytest.mark.parametrize("rank", [3, 1])
def test_ged_ssd_runs(bandpass_filter: bool, rank: int) -> None:
    """Check that SpatioSpectralFilter class runs SSD."""
    n_chans = 3
    n_epochs = 5
    n_times = 500
    sampling_freq = 100
    data = _generate_data((n_epochs, n_chans, n_times), complexobj=False)
    signal_bounds = (10, 15)
    noise_bounds = (8, 17)

    ssf = SpatioSpectralFilter(data, sampling_freq)

    transformed_data = ssf.fit_transform_ssd(
        signal_bounds=signal_bounds,
        noise_bounds=noise_bounds,
        bandpass_filter=bandpass_filter,
        rank=rank,
    )

    assert isinstance(transformed_data, np.ndarray), (
        "`transformed_data` should be a NumPy array."
    )
    assert transformed_data.shape == (
        n_epochs,
        rank,
        n_times,
    ), "`transformed_data` should have shape (n_epochs, rank, n_times)."
    assert np.allclose(transformed_data, ssf.get_transformed_data(), rtol=0, atol=0), (
        "data returned from `fit_transform_ssd()` and `get_transformed_data()` should "
        "be identical."
    )

    if rank > 1:
        transformed_data = ssf.get_transformed_data(
            min_ratio=ssf.ratios[-2], copy=False
        )
        assert transformed_data.shape == (
            n_epochs,
            rank - 1,
            n_times,
        ), "`transformed_data` should have shape (n_epochs, rank - 1, n_times)."

    assert isinstance(ssf.filters, np.ndarray), "`filters` should be a NumPy array."
    assert ssf.filters.shape == (
        n_chans,
        rank,
    ), "`filters` should have shape (n_chans, rank)."

    assert isinstance(ssf.patterns, np.ndarray), "`patterns` should be a NumPy array."
    assert ssf.patterns.shape == (
        rank,
        n_chans,
    ), "`patterns` should have shape (rank, n_chans)."

    assert isinstance(ssf.ratios, np.ndarray), "`ratios` should be a NumPy array."
    assert ssf.ratios.shape == (rank,), "`ratios` should have shape (rank)."

    empty_transformed_data = ssf.get_transformed_data(min_ratio=ssf.ratios.max() + 1)
    assert empty_transformed_data.size == 0, (
        "`transformed_data` should be empty when too high a ratio is requested."
    )


@pytest.mark.filterwarnings(
    r"ignore:No signal\-to\-noise ratios are greater than the requested minimum:"
    "UserWarning"
)
@pytest.mark.parametrize("csd_method", ["fourier", "multitaper"])
@pytest.mark.parametrize("rank", [3, 1])
def test_ged_hpmax_runs(csd_method: str, rank: int) -> None:
    """Check that SpatioSpectralFilter class runs HPMax."""
    n_chans = 3
    n_epochs = 5
    n_times = 100
    sampling_freq = 100
    data = _generate_data((n_epochs, n_chans, n_times), complexobj=False)
    signal_bounds = (10, 15)
    noise_bounds = (8, 17)

    ssf = SpatioSpectralFilter(data, sampling_freq)
    transformed_data = ssf.fit_transform_hpmax(
        signal_bounds=signal_bounds,
        noise_bounds=noise_bounds,
        rank=rank,
        csd_method=csd_method,
    )

    assert isinstance(transformed_data, np.ndarray), (
        "`transformed_data` should be a NumPy array."
    )
    assert transformed_data.shape == (
        n_epochs,
        rank,
        n_times,
    ), "`transformed_data` should have shape (n_epochs, rank, n_times)."
    assert np.allclose(transformed_data, ssf.get_transformed_data(), rtol=0, atol=0), (
        "data returned from `fit_transform_hpmax()` and `get_transformed_data()` "
        "should be identical."
    )

    assert isinstance(ssf.filters, np.ndarray), "`filters` should be a NumPy array."
    assert ssf.filters.shape == (
        n_chans,
        rank,
    ), "`filters` should have shape (n_chans, rank)."

    assert isinstance(ssf.patterns, np.ndarray), "`patterns` should be a NumPy array."
    assert ssf.patterns.shape == (
        rank,
        n_chans,
    ), "`patterns` should have shape (rank, n_chans)."

    assert isinstance(ssf.ratios, np.ndarray), "`ratios` should be a NumPy array."
    assert ssf.ratios.shape == (rank,), "`ratios` should have shape (rank)."

    empty_transformed_data = ssf.get_transformed_data(min_ratio=ssf.ratios.max() + 1)
    assert empty_transformed_data.size == 0, (
        "`transformed_data` should be empty when too high a ratio is requested."
    )

    # test it works with parallelisation
    ssf.fit_transform_hpmax(
        signal_bounds=signal_bounds,
        noise_bounds=noise_bounds,
        rank=rank,
        csd_method=csd_method,
        n_jobs=2,
    )
    ssf.fit_transform_hpmax(
        signal_bounds=signal_bounds,
        noise_bounds=noise_bounds,
        rank=rank,
        csd_method=csd_method,
        n_jobs=-1,
    )
