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
A Python signal processing package for computing spectral-domain interactions
using bispectra.

This package provides the tools for computing spectral-domain interactions
between signals such as phase-amplitude coupling (PAC) based on bispectra, and
phase-phase coupling (PPC). Parallel processing and
`Numba <https://numba.pydata.org/>`_ optimisation are implemented to reduce
computation times. Additional tools for plotting results and computing Fourier
coefficients of data are also provided.

.. toctree::
   :maxdepth: 4
   :titlesonly:
   :caption: Contents:

   motivation
   installation
   examples
   api
