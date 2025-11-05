# PyBispectra Changelog

## [Version 1.3.0dev](https://pybispectra.readthedocs.io/latest/)

##### Enhancements
- Added support for computing time-resolved bispectral features for the `PAC`, `WaveShape`, `General`, and `Threenorm` classes.
- Added `times` parameter to the `compute()` methods of the `PAC`, `WaveShape`, `General`, and `Threenorm` classes to specify time windows for computing time-resolved features over.
- Added support for storing and plotting time-resolved results in the `ResultsCFC`, `ResultsWaveShape`, and `ResultsGeneral` classes.
- Added `norm` parameter to `WaveShape.compute()` to control normalisation of waveshape results.
- Added `output` parameter to `compute_tfr()` to allow complex coefficients to be returned.

##### Bug Fixes
- Fixed error where the coupling with `PPC` was not being computed correctly.

##### API
- Changed the `data` parameter of `PPC` to require time-frequency representations instead of non-time-resolved Fourier coefficients.

##### Documentation
- Added a new example for computing time-resolved bispectral features.

<br>

## [Version 1.2](https://pybispectra.readthedocs.io/1.2/)

### Version 1.2.4

##### Bug Fixes
- Fixed error where univariate normalisation of antisymmetrised PAC was not being applied correctly.

<br>

### Version 1.2.3

##### Bug Fixes
- Fixed error where NumPy integers and floats were not being recognised as valid types.

<br>

### Version 1.2.2

##### Bug Fixes
- Fixed error where `indices` in `ResultsCFC`, `ResultsTDE`, and `ResultsGeneral` classes were not being mapped to results correctly.

##### Documentation
- Improved the warning about invalid frequency combinations returning `np.nan` values.

<br>

### Version 1.2.1

##### Dependencies
- Added `scikit-learn` as a dependency for compatibility with `mne>=1.9`.

<br>

### Version 1.2.0

##### Enhancements
- Added general `Bispectrum` and `Threenorm` classes for computing with flexible kmn channel combinations.
- Added the option to control whether a copy is returned from the `get_results()` method of all `Results...` classes and from `SpatioSpectralFilter.get_transformed_data()` (default behaviour returns a copy, like in previous versions).
- Added new `fit_ssd()`, `fit_hpmax()`, and `transform()` methods to the `SpatioSpectralFilter` class to bring it more in line with `scikit-learn` fit-transform classes.

##### Bug Fixes
- Fixed error where the number of subplots exceeding the number of nodes would cause plotting to fail.
- Fixed error where bandpass filter settings for the SSD method in `SpatioSpectralFilter` were not being applied correctly.

##### API
- Changed the default value of `min_ratio` in `SpatioSpectralFilter.get_transformed_data()` from `1.0` to `-inf`.

##### Documentation
- Added a new example for computing the bispectrum and threenorm using the general classes.

<br>

## [Version 1.1](https://pybispectra.readthedocs.io/1.1/)

### Version 1.1.0

##### Enhancements
- Reduced the memory requirement of bispectrum computations.
- Added support for computing & storing time delays of multiple frequency bands simultaneously.
- Added a new option for controlling the colour bar of waveshape plots.
- Added an option for controlling the precision of computations.

##### Bug Fixes
- Fixed incorrect channel indexing for time delay antisymmetrisation.

##### API
- Changed how operations on specific frequency/time ranges are specified to be more flexible.

##### Documentation
- Added a new example for computing time delays on specific frequency bands.

<br>

## [Version 1.0](https://pybispectra.readthedocs.io/1.0/)

### Version 1.0.0

- Initial release.