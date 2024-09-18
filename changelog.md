# PyBispectra Changelog

## [Version 1.2.1dev](https://pybispectra.readthedocs.io/en/main/)

## [Version 1.2.0](https://pybispectra.readthedocs.io/en/1.2.0/)

##### Enhancements
- Added general `Bispectrum` and `Threenorm` classes for computing with flexible kmn channel combinations.

##### Bug Fixes
- Fixed error where the number of subplots exceeding the number of nodes would cause plotting to fail.
- Fixed error where bandpass filter settings for the SSD method in `SpatioSpectralFilter` were not being applied correctly.

##### API
- Changed the default value of `min_ratio` in `SpatioSpectralFilter.get_transformed_data()` from `1.0` to `-inf`.
- Added the option to control whether a copy is returned from the `get_results()` method of all `Results...` classes and from `SpatioSpectralFilter.get_transformed_data()` (default behaviour returns a copy, like in previous versions).
- Added new `fit_ssd()`, `fit_hpmax()`, and `transform()` methods to the `SpatioSpectralFilter` class to bring it more in line with `scikit-learn` fit-transform classes.

##### Documentation
- Added a new example for computing the bispectrum and threenorm using the general classes.

## [Version 1.1.0](https://pybispectra.readthedocs.io/en/1.1.0/)

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


## [Version 1.0.0](https://pybispectra.readthedocs.io/en/1.0.0/)

- Initial release.