.. PyBispectra documentation master file, created by
   sphinx-quickstart on Sat Mar 11 17:27:35 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. title:: Home

.. define new line for html
.. |br| raw:: html

   <br />

.. image:: https://tsbinns.com/assets/pybispectra/logo.gif
   :alt: PyBispectra

|br|
A Python signal processing package for computing spectral-domain and
time-domain interactions using bispectra.

This package provides the tools for performing spectral- and time-domain
analyses based on bispectra, including phase-amplitude coupling, time delay
estimation, and wave shape features. Additional tools for computing phase-phase
coupling, generalised eigendecompositions, Fourier coefficients, and plotting
results are also provided.

Parallel processing and `Numba <https://numba.pydata.org/>`_ optimisation are
implemented to reduce computation times. There is a minor reliance on
`MNE <https://mne.tools/stable/index.html>`_.

.. toctree::
   :maxdepth: 4
   :titlesonly:
   :caption: Contents:

   motivation
   installation
   examples
   api
