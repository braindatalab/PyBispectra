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
# Generating data and computing Fourier coefficients
# --------------------------------------------------
#
# We will start by generating some data that we can compute PPC on, then
# compute the Fourier coefficients of the data.

# %%

# generate data
random = np.random.RandomState(44)
data = random.rand(30, 2, 500)  # [epochs x channels x frequencies]
sfreq = 100  # sampling frequency in Hz

# compute Fourier coeffs.
fft, freqs = compute_fft(data=data, sfreq=sfreq)

print(
    f"FFT coeffs.: [{fft.shape[0]} epochs x {fft.shape[1]} channels x "
    f"{fft.shape[2]} frequencies]\nFreq. range: {freqs[0]} - {freqs[-1]} Hz"
)

###############################################################################
# As you can see, we have FFT coefficients for 2 channels across 30 epochs,
# with 101 frequencies ranging from 0 to 50 Hz with a frequency resolution of
# 0.5 Hz. We will use these coefficients to compute PPC.


###############################################################################
# References
# -----------------------------------------------------------------------------
# .. footbibliography::
