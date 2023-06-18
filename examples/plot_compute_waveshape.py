"""
===========================
Compute wave shape features
===========================

This example demonstrates how wave shape features can be computed with
PyBispectra.
"""

# %%

import numpy as np

from pybispectra import compute_fft, WaveShape

###############################################################################
# Background
# ----------
# When analysing signals, important information may be gleaned from a variety
# of features, including the shape of the wave forms. For example, in
# neuroscience it has been suggested that non-sinusoidal waves may play
# important roles in physiology and pathology, such as waveform sharpness
# reflecting synchrony of synaptic inputs :footcite:`Sherman2016` and
# correlating with symptoms of Parkinson's disease :footcite:`Cole2017`. Two
# aspects of waveform shape described in recent literature include: rise-decay
# asymmetry - how much the wave resembles a sawtooth pattern (also called
# waveform sharpness or derivative skewness); and peak-trough asymmetry -
# whether peaks (events with a positive-valued amplitude) or troughs (events
# with a negative-valued amplitude) are more dominant in the signal (also
# called signal/value skewness).
#
# A common strategy for wave shape analysis involves identifying and
# characterising the features of waves in time-series data - see Cole *et al.*
# (2017) :footcite:`Cole2017` for an example. Naturally, it can be of interest
# to explore the waveform shapes of signals at particular frequencies, in which
# case the time-series data can be bandpass filtered. There is, however, a
# major limitation to this approach: applying a bandpass filter to data can
# seriously alter the waveform shape at the filtered frequencies, compromising
# any analysis of waveform shape before it has begun.
#
# Thankfully, the bispectrum captures information about waveform shape,
# enabling spectrally-resolved analyses at a fine frequency resolution without
# any need for bandpass filtering. In particular, the bispectrum contains
# information about rise-decay asymmetry (encoded in the imaginary part of the
# bispectrum) and peak-trough asymmetry (encoded in the real part of the
# bispectrum) :footcite:`Bartz2019`.
#
# The bispectrum, :math:`B`, has the general form:
#
# :math:`\large B_{kmn}(f_1,f_2)=<\vec{k}(f_1)\vec{m}(f_2)\vec{n}^*(f_2+f_1)>`,
#
# where :math:`kmn` is a combination of the fourier coefficients of channels
# :math:`\vec{x}` and :math:`\vec{y}`, :math:`f` represents a given frequency,
# and the angled brackets represent the averaged value over epochs. When
# analysing waveform shape, we are interested in only a single signal, and as
# such :math:`k=m=n`.
#
# Furthermore, we can normalise the bispectrum to the bicoherence,
# :math:`\mathcal{B}` whose values lie in the range :math:`[-1, 1]`. This
# normalisation can be performed using the threenorm, :math:`N`
# :footcite:`Zandvoort2021`:
#
# :math:`\large N_{xyy}(f_1,f_2)=(<|\vec{x}(f_1)|^3><|\vec{y}(f_2)|^3>
# <|\vec{y}(f_2+f_1)|^3>)^{\frac{1}{3}}` ,
#
# :math:`\large \mathcal{B}_{xyy}(f_1,f_2)=\Large
# \frac{B_{xyy}(f_1,f_2)}{N_{xyy}(f_1,f_2)}` .

###############################################################################
# Generating data and computing Fourier coefficients
# --------------------------------------------------
# We will start by loading some example data that we can compute PAC on, then
# compute the Fourier coefficients of the data.

# %%

# load example data
data = np.load("example_data_cfc.npy")  # [epochs x channels x frequencies]
sampling_freq = 200  # Hz

# compute Fourier coeffs.
fft, freqs = compute_fft(
    data=data, sampling_freq=sampling_freq, n_points=sampling_freq
)

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

waveshape = WaveShape(
    data=fft, freqs=freqs, sampling_freq=sampling_freq
)  # initialise object
waveshape.compute(indices=tuple([0]))  # compute PAC

waveshape_results = waveshape.results.get_results()  # return results as array

print(
    f"Wave shape results: [{waveshape_results.shape[0]} channels x "
    f"{waveshape_results.shape[1]} f1s x {waveshape_results.shape[2]} f2s]"
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

fig, axes = waveshape.results.plot(
    f1s=np.arange(0, 51),
    major_tick_intervals=10.0,
    minor_tick_intervals=2.0,
)

print("jeff")

###############################################################################
# References
# -----------------------------------------------------------------------------
# .. footbibliography::

# %%
