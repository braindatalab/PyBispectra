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
# :math:`\large PPC(x_{f_1}, y_{f_2})=\LARGE \frac{|\langle A_x(f_1)A_y(f_2) e^{i(\varphi_x(f_1)\frac{f_2}{f_1}-\varphi_x(f_2))} \rangle|}{\langle A_x(f_1)A_y(f_2) \rangle}`,  # noqa E501
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
#
# Computing PPC
# -------------
#
# To compute PPC, we start by initialising the :class:`PPC` class object with
# the FFT coefficients and the frequency information. To compute PPC, we call
# the :meth:`compute` method. By default, PPC is computed between all channel
# and frequency combinations, however we can also specify particular
# combinations of interest.
#
# Here, we specify the :attr:`indices` to compute PPC on. :attr:`indices` is
# expected to be a tuple containing two NumPy arrays for the indices of the
# seed and target channels, respectively. The indices specified below mean that
# PPC will only be computed across frequencies within each channel (i.e.
# 0 -> 0; and 1 -> 1). By leaving the frequency arguments :attr:`f1` and
# :attr:`f2` blank, we will look at all possible frequency combinations.

# %%

ppc = PPC(data=fft, freqs=freqs)  # initialise object
ppc.compute(
    indices=(np.array([0, 1]), np.array([0, 1])), f1=None, f2=None
)  # compute PPC

ppc_results = ppc.results.get_results()  # return results

print(
    f"PPC results: [{ppc_results.shape[0]} connections x "
    f"{ppc_results.shape[1]} f1 x {ppc_results.shape[2]} f2]"
)

###############################################################################
# We can see that PPC has been computed for 2 connections (0 -> 0; and 1 -> 1),
# and all possible frequency combinations, averaged across our 30 epochs.
# Whilst there are 10,000 such frequency combinations in our [100 x 100]
# matrices, PPC for those entries where :math:`f1` would be higher than
# :math:`f2` cannot be computed, in which case the values are ``numpy.nan``
# (see the plotted results below for a visual demonstration of this).


###############################################################################
# References
# -----------------------------------------------------------------------------
# .. footbibliography::
