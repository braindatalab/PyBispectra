![PyBispectra logo](docs/source/_static/logo.gif)

A Python signal processing package for computing spectral- and time-domain interactions using the bispectrum.

This package provides the tools for computing phase-amplitude coupling, time delay estimation, and wave shape features using the bispectrum and bicoherence. Additional tools for computing amplitude-amplitude coupling, phase-phase coupling, and spatio-spectral filters are also provided.

Parallel processing and [Numba](https://numba.pydata.org/) optimisation are implemented to reduce computation times.

Analysis of phase-amplitude coupling, time delays, and non-sinusoidal waveshape provide important insights into electrophysiology data, but traditional analysis methods have critical limitations. In contrast, the bispectrum - the Fourier transform of the third order moment - offers approaches to perform such analyses whilst overcoming many of the limitations of traditional methods.

## Installation & Requirements:
Install the package into the desired environment using pip: `pip install pybispectra`<br/>
or conda: `conda install -c conda-forge pybispectra`<br/>
More information on the [installation](https://pybispectra.readthedocs.io/latest/installation.html) page.

## Use:
To get started with the toolbox, check out the [documentation](https://pybispectra.readthedocs.io/latest/) and [examples](https://pybispectra.readthedocs.io/latest/examples.html).

For instance, given some epoched time series, `data`, phase-amplitude coupling can be computed as:

```python
from pybispectra import PAC, compute_fft

coeffs, freqs = compute_fft(data, sampling_freq)  # compute spectral coeffs
pac = PAC(coeffs, freqs, sampling_freq)  # initialise coupling object
pac.compute()  # compute phase-amplitude coupling
pac_results = pac.results  # extract results
pac_results.plot()  # plot results
```

## Contributing & Development:
If you encounter issues with the package, want to suggest improvements, or have made any changes which you would like to see officially supported, please refer to the [development](https://pybispectra.readthedocs.io/latest/development.html) page. A unit test suite is included and must be expanded where necessary to validate any changes.

## Citing:
If you use this toolbox in your work, please include the following citation:<br/>
Binns, TS, Pellegrini, F, Jurhar, T, Nguyen, TD, Köhler, RM, & Haufe, S (2025). PyBispectra: A toolbox for advanced electrophysiological signal processing using the bispectrum. *Journal of Open Source Software*. DOI: [10.21105/joss.08504](https://doi.org/10.21105/joss.08504)
