.. PyBispectra documentation master file

.. title:: Home

.. define new line for html
.. |br| raw:: html

   <br />

.. image:: _static/logo.gif
   :alt: PyBispectra

|br|
A Python signal processing package for computing spectral- and time-domain interactions
using the bispectrum.

This package provides the tools for computing phase-amplitude coupling, time delay
estimation, and waveshape features using the bispectrum and bicoherence. Additional
tools for computing amplitude-amplitude coupling, phase-phase coupling, and
spatio-spectral filters are also provided.

Parallel processing and `Numba <https://numba.pydata.org/>`_ optimisation are
implemented to reduce computation times.

If you use this toolbox in your work, please include the following citation:|br|
Binns, TS, Pellegrini, F, Jurhar, T, Nguyen, TD, Köhler, RM, & Haufe, S (2025).
PyBispectra: A toolbox for advanced electrophysiological signal processing using the
bispectrum. *Journal of Open Source Software*. DOI:
`10.21105/joss.08504 <https://doi.org/10.21105/joss.08504>`_

.. toctree::
   :maxdepth: 2
   :titlesonly:
   :caption: Contents:

   motivation
   installation
   examples
   api
   development
