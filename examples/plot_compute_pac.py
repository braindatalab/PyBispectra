"""
================================
Compute phase-amplitude coupling
================================

This example demonstrates how phase-amplitude coupling (PAC) can be computed
with PyBispectra.
"""

# %%

import numpy as np

from pybispectra import compute_fft, PAC

###############################################################################
# Background
# ----------
# PAC quantifies the relationship between the phases of a lower frequency
# :math:`f_1` and the amplitude of a higher frequency :math:`f_2` within a
# single signal, :math:`vec{x}`, or across different signals, :math:`vec{x}`
# and :math:`vec{y}`.
#
# The method available in PyBispectra is based on the bispectrum, :math:`B`,
# with four variations available. The bispectrum has the general form:
#
# :math:`\large B_{kmn}(f_1,f_2)=<\vec{k}(f_1)\vec{m}(f_2)\vec{n}^*(f_2+f_1)>`,
#
# where :math:`kmn` is a combination of channels :math:`\vec{x}` and
# :math:`\vec{y}`, and the angled brackets represent the averaged value over
# epochs. The computation of PAC follows from this :footcite:`Kovach2018`:
#
# :math:`\large PAC(\vec{x}_{f_1},\vec{y}_{f_2})=B_{xyy}(f_1)B_{xyy}(f_2)B_{xyy}^*(f_2+f_1)`.
#
# The four variations arise from the options for normalisation and
# antisymmetrisation. The bispectrum can be normalised to the bicoherence,
# :math:`\mathcal{B}`, using the threenorm, :math:`N`
# :footcite:`Zandvoort2021`:
#
# :math:`\large N_{xyy}(f_1,f_2)=(<|\vec{x}(f_1)|^3><|\vec{y}(f_2)|^3><|\vec{y}(f_2+f_1)|^3>)^{\frac{1}{3}}`,
#
# :math:`\large \mathcal{B}_{xyy}(f_1,f_2)=\Large \frac{B_{xyy}(f_1,f_2)}{N_{xyy}(f_1,f_2)}`,
#
# where the resulting PAC results are this normalised by the power of the
# corresponding frequencies. Furthermore, PAC can be antisymmetrised by
# subtracting the results from those found using the transposed bispectrum,
# :math:`B_{xyx}` :footcite:`Chella2014`. Antisymmetrisation allows you to
# reduce spurious coupling resulting from artefacts of volume conduction,
# giving you a more robust connectivty metric.

###############################################################################
# Generating data and computing Fourier coefficients
# --------------------------------------------------
# We will start by generating some data that we can compute PAC on, then
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
# 0.5 Hz. We will use these coefficients to compute PAC.
#
# Computing PAC
# -------------
# To compute PAC, we start by initialising the :class:`PAC` class object with
# the FFT coefficients and the frequency information. To compute PAC, we call
# the :meth:`compute` method. By default, PAC is computed between all channel
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

pac = PAC(data=fft, freqs=freqs)  # initialise object
pac.compute(indices=None, f1=np.arange(5, 10), f2=(15, 25))  # compute PAC

pac_results = pac.results.get_results()  # return results as array

print(
    f"PPC results: [{pac_results.shape[0]} connections x "
    f"{pac_results.shape[1]} f1 x {pac_results.shape[2]} f2]"
)

###############################################################################
# We can see that PAC has been computed for 2 connections (0 -> 0; and 1 -> 1),
# and all possible frequency combinations, averaged across our 30 epochs.
# Whilst there are 10,000 such frequency combinations in our [100 x 100]
# matrices, PPC for those entries where :math:`f1` would be higher than
# :math:`f2` cannot be computed, in which case the values are ``numpy.nan``
# (see the plotted results below for a visual demonstration of this).

###############################################################################
# Plotting PPC
# ------------
# Let us now inspect the results. For this, we will plot the results for both
# connections on the same plot. If we wished, we could plot this information on
# separate plots, or specify a subset of frequencies to inspect.

# %%

fig, axes = pac.results.plot(n_rows=1, n_cols=2)  # 2 subplots for the cons.

###############################################################################
# As you can see, values for the lower right triangle of each plot are missing,
# corresponding to the frequency combinations where :math:`f_1` is greater than
# :math:`f_2`, and hence where PPC cannot be computed. Note that the ``Figure``
# and ``Axes`` objects can also be returned for any desired manual adjustments
# of the plots.
#
# Controlling for spurious PAC with PPC
# -------------------------------------
# Now that we have an idea of how PAC and PPC can be computed, the following
# example will look at how PPC can be used to control for spurious PAC results
# stemming from frequency harmonics :footcite:`Giehl2021`.

###############################################################################
# References
# -----------------------------------------------------------------------------
# .. footbibliography::
