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
# :math:`\large PAC(\vec{x}_{f_1},\vec{y}_{f_2})=B_{xyy}(f_1)B_{xyy}(f_2)
# B_{xyy}^*(f_2+f_1)`.
#
# The four variations arise from the options for normalisation and
# antisymmetrisation. The bispectrum can be normalised to the bicoherence,
# :math:`\mathcal{B}`, using the threenorm, :math:`N`
# :footcite:`Zandvoort2021`:
#
# :math:`\large N_{xyy}(f_1,f_2)=(<|\vec{x}(f_1)|^3><|\vec{y}(f_2)|^3>
# <|\vec{y}(f_2+f_1)|^3>)^{\frac{1}{3}}`,
#
# :math:`\large \mathcal{B}_{xyy}(f_1,f_2)=\Large
# \frac{B_{xyy}(f_1,f_2)}{N_{xyy}(f_1,f_2)}`,
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
# We will start by loading some example data that we can compute PAC on, then
# compute the Fourier coefficients of the data.

# %%

# load example data
data = np.load("example_data_cfc.npy")  # [epochs x channels x frequencies]
sfreq = 200  # sampling frequency in Hz

# compute Fourier coeffs.
fft, freqs = compute_fft(data=data, sampling_freq=sfreq, n_points=sfreq)

print(
    f"FFT coeffs.: [{fft.shape[0]} epochs x {fft.shape[1]} channels x "
    f"{fft.shape[2]} frequencies]\nFreq. range: {freqs[0]} - {freqs[-1]} Hz"
)

###############################################################################
# As you can see, we have FFT coefficients for 2 channels across 30 epochs,
# with 101 frequencies ranging from 0 to 100 Hz with a frequency resolution of
# 1 Hz. We will use these coefficients to compute PAC.
#
# Computing PAC
# -------------
# To compute PAC, we start by initialising the :class:`PAC` class object with
# the FFT coefficients and the frequency information. To compute PAC, we call
# the :meth:`compute` method. By default, PAC is computed between all channel
# and frequency combinations, however we can also specify particular
# combinations of interest.
#
# Here, we specify the :attr:`indices` to compute PAC on. :attr:`indices` is
# expected to be a tuple containing two NumPy arrays for the indices of the
# seed and target channels, respectively. The indices specified below mean that
# PAC will only be computed across frequencies between the channels (i.e.
# 0 -> 1). By leaving the frequency arguments :attr:`f1s` and :attr:`f2s`
# blank, we will look at all possible frequency combinations.

# %%

pac = PAC(data=fft, freqs=freqs, sampling_freq=sfreq)  # initialise object
pac.compute(indices=([0], [1]))  # compute PAC

pac_results = pac.results[0].get_results()  # return results as array

print(
    f"PPC results: [{pac_results.shape[0]} connections x "
    f"{pac_results.shape[1]} f1s x {pac_results.shape[2]} f2s]"
)

###############################################################################
# We can see that PAC has been computed for 2 connections (0 -> 0; and 1 -> 1),
# and all possible frequency combinations, averaged across our 30 epochs.
# Whilst there are > 10,000 such frequency combinations in our [101 x 101]
# matrices, PAC for those entries where :math:`f_1` would be higher than
# :math:`f_2`, as well as where :math:`f_2 + f_1` exceeds the frequency bounds
# of our data, cannot be computed. In such cases, the values are ``numpy.nan``
# (see the plotted results below for a visual demonstration of this).

###############################################################################
# Plotting PAC
# ------------
# Let us now inspect the results. For this, we will plot the results for both
# connections on the same plot. If we wished, we could plot this information on
# separate plots, or specify a subset of frequencies to inspect. Note that the
# ``Figure`` and ``Axes`` objects can also be returned for any desired manual
# adjustments of the plots.

# %%

fig, axes = pac.results[0].plot(
    f1s=np.arange(0, 51), major_tick_intervals=10.0, minor_tick_intervals=2.0
)

###############################################################################
# References
# -----------------------------------------------------------------------------
# .. footbibliography::

# %%
