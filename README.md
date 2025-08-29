![](docs/source/_static/logo.gif)

A Python signal processing package for computing spectral- and time-domain
interactions using the bispectrum.

This package provides the tools for computing phase-amplitude coupling, time
delay estimation, and wave shape features using the bispectrum and bicoherence.
Additional tools for computing amplitude-amplitude coupling, phase-phase
coupling, and spatio-spectral filters are also provided.

Parallel processing and [Numba](https://numba.pydata.org/) optimisation are
implemented to reduce computation times. There is a minor reliance on the
[MNE](https://mne.tools/stable/index.html) signal processing toolbox.

## Installation & Requirements:
Install the package into the desired environment using pip `pip install pybispectra`<br/>
[See here for the list of requirements](requirements.txt).

## Use:
To get started with the toolbox, check out the [documentation](https://pybispectra.readthedocs.io/en/1.0.0/) and [examples](https://pybispectra.readthedocs.io/en/1.0.0/examples.html).

## Citing:
If you use this toolbox in your work, please include the following citation:<br/>
Binns, TS, Pellegrini, F, Jurhar, T, Nguyen, TD, Köhler, RM, & Haufe, S (2025). PyBispectra: A toolbox for advanced electrophysiological signal processing using the bispectrum. *Journal of Open Source Software*. DOI: [10.21105/joss.08504](https://doi.org/10.21105/joss.08504)
