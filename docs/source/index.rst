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
delay estimation, and wave shape features using the bispectrum and bicoherence.
Additional tools for computing amplitude-amplitude coupling, phase-phase
coupling, and spatio-spectral filters are also provided.

Parallel processing and `Numba <https://numba.pydata.org/>`_ optimisation are
implemented to reduce computation times. There is a minor reliance on the
`MNE <https://mne.tools/stable/index.html>`_ signal processing toolbox.

.. toctree::
   :maxdepth: 2
   :titlesonly:
   :caption: Contents:

   motivation
   installation
   examples
   api
   development
