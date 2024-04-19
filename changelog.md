# PyBispectra Changelog

## [Version 1.2.0dev](https://pybispectra.readthedocs.io/en/main/)

##### Enhancements
- Added general classes for computing the bispectrum and threenorm with flexible kmn channel combinations.

##### Bug Fixes
- Fixed error where the number of subplots exceeding the number of nodes would cause plotting to fail.

##### Documentation
- Added a new example for computing the bispectrum and threenorm using the general classes.

## [Version 1.1.0](https://pybispectra.readthedocs.io/en/main/)

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