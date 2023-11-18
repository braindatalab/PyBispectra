.. PyBispectra documentation master file

.. title:: Home

.. define new line for html
.. |br| raw:: html

   <br />

.. image:: _static/logo.gif
   :alt: PyBispectra

|br|
A Python signal processing package for computing spectral- and time-domain
interactions using the bispectrum.

This package provides the tools for computing phase-amplitude coupling, time
delay estimation, and waveshape features using the bispectrum and bicoherence.
Additional tools for computing amplitude-amplitude coupling, phase-phase
coupling, and spatio-spectral filters are also provided.

Parallel processing and `Numba <https://numba.pydata.org/>`_ optimisation are
implemented to reduce computation times. There is a minor reliance on the
`MNE <https://mne.tools/stable/index.html>`_ signal processing toolbox.

If you use this toolbox in your work, please include the following citation:|br|
Binns, T. S., Pellegrini, F., Jurhar, T., & Haufe, S. (2023). PyBispectra (Version 1.1.0). DOI: `10.5281/zenodo.10155044 <https://doi.org/10.5281/zenodo.10155044>`_

.. toctree::
   :maxdepth: 2
   :titlesonly:
   :caption: Contents:

   motivation
   installation
   examples
   api
   development
