"""
============================
Compute phase-phase coupling
============================

This example demonstrates how phase-phase coupling (PPC) can be computed with
PyBispectra.
"""

# %%

import numpy as np

from pybispectra import compute_fft, PPC

###############################################################################
# Background
# ----------
#
# PPC quantifies the relationship between the phases of a lower frequency
# :math:`f_1` and a higher frequency :math:`f_2` within a single signal, or
# across different signals.
#
# The method available in PyBispectra can be thought of as a measure of
# coherence between frequencies :footcite:`Giehl2021` (note that it is not
# based on bispectra):
#
# :math:`\large PPC(x_{f_1}, y_{f_2})=\LARGE \frac{|\langle A_x(f_1)A_y(f_2) e^{i(\varphi_x(f_1)\frac{f_2}{f_1}-\varphi_x(f_2))} \rangle|}{\langle A_x(f_1)A_y(f_2) \rangle}`,
#
# where :math:`A(f)` and :math:`\varphi(f)` are the amplitude and phase of a
# signal at a given frequency, respectively, and the angled brackets represent
# the average over epochs. The phase of :math:`f_1` is accelerated to match
# that of :math:`f_2` by scaling the phase by a factor of
# :math:`\frac{f_2}{f_1}`. PPC values for this measure lie in the range
# :math:`[0, 1]`, with 0 representing a random phase relationship, and 1
# representing perfect phase coupling.


###############################################################################
# References
# -----------------------------------------------------------------------------
# .. footbibliography::
