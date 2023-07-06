"""
===========================
Compute wave shape features
===========================

This example demonstrates how wave shape features can be computed with
PyBispectra.
"""

# %%

import os

import numpy as np
from matplotlib import pyplot as plt

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
# We will start by loading some example data and computing the Fourier
# coefficients using the :func:`~pybispectra.utils.compute_fft` function. This
# data consists of sawtooth waves (information will be captured in the
# rise-decay asymmetry) and waves with a dominance of peaks or troughs
# (information will be captured in the peak-trough asymmetry), all simulated at
# 20 Hz.

# %%

# load example data
data_sawtooths = np.load(
    os.path.join("data", "sim_data_waveshape_sawtooths.npy")
)
data_peaks_troughs = np.load(
    os.path.join("data", "sim_data_waveshape_peaks_troughs.npy")
)
sampling_freq = 200  # Hz

# plot timeseries data
times = np.linspace(
    0,
    (data_sawtooths.shape[2] / sampling_freq) / 4,
    data_sawtooths.shape[2] // 4,
)
fig, axes = plt.subplots(2, 2)
axes[0, 0].plot(times, data_sawtooths[0, 0, : times.shape[0]])
axes[0, 1].plot(times, data_sawtooths[0, 1, : times.shape[0]])
axes[1, 0].plot(times, data_peaks_troughs[0, 0, : times.shape[0]])
axes[1, 1].plot(times, data_peaks_troughs[0, 1, : times.shape[0]])
titles = [
    "Steepness: rise > decay",
    "Steepness: decay > rise",
    "Dominance: peaks",
    "Dominance: troughs",
]
for ax, title in zip(axes.flatten(), titles):
    ax.set_title(title)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude (A.U.)")
fig.tight_layout()

# compute Fourier coeffs.
fft_coeffs_sawtooths, freqs = compute_fft(
    data=data_sawtooths, sampling_freq=sampling_freq, n_points=sampling_freq
)
fft_coeffs_peaks_troughs, _ = compute_fft(
    data=data_peaks_troughs,
    sampling_freq=sampling_freq,
    n_points=sampling_freq,
)

###############################################################################
# Plotting the data, we see that the sawtooth waves consist of a signal where
# the rise steepness is greater than the decay steepness and a signal where the
# decay steepness is greater than the rise steepness. Additionally, the peak
# and trough waves consist of a signal where peaks are most dominant, and a
# signal where troughs are most dominant.
#
# Computing wave shape features
# -----------------------------
# To compute wave shape, we start by initialising the
# :class:`~pybispectra.waveshape.WaveShape` class object with the FFT
# coefficients and the frequency information. To compute wave shape, we call
# the :meth:`~pybispectra.waveshape.WaveShape.compute` method. By default, wave
# shape is computed for all channels and all frequency combinations, however we
# can also specify particular channels and combinations of interest.
#
# Here, we specify the frequency arguments
# :attr:`~pybispectra.waveshape.WaveShape.f1s` and
# :attr:`~pybispectra.waveshape.WaveShape.f2s` to compute wave shape on in the
# range 15-25 Hz (around the frequency at which the signal features were
# simulated). By leaving the indices argument blank, we will look at all
# channels in the data.

# %%

# sawtooth waves
waveshape_sawtooths = WaveShape(
    data=fft_coeffs_sawtooths, freqs=freqs, sampling_freq=sampling_freq
)  # initialise object
waveshape_sawtooths.compute()  # compute wave shape

# peaks and troughs
waveshape_peaks_troughs = WaveShape(
    data=fft_coeffs_peaks_troughs, freqs=freqs, sampling_freq=sampling_freq
)
waveshape_peaks_troughs.compute()

# return results as an array
waveshape_results = waveshape_sawtooths.results.get_results()

print(
    f"Wave shape results: [{waveshape_results.shape[0]} channels x "
    f"{waveshape_results.shape[1]} f1s x {waveshape_results.shape[2]} f2s]"
)

###############################################################################
# We can see that wave shape features have been computed for both channels and
# the specified frequency combinations, averaged across our 5 epochs. Given the
# nature of the bispectrum, entries where :math:`f_1` would be higher than
# :math:`f_2`, as well as where :math:`f_2 + f_1` exceeds the frequency bounds
# of our data, cannot be computed. Although this does not apply here given the
# limited frequency ranges, in such cases, the values corresponding to those
# 'bad' frequency combinations are ``numpy.nan``.

###############################################################################
# Plotting wave shape features
# ----------------------------
# Let us now inspect the results. For this, we will plot the results for both
# connections on the same plot. If we wished, we could plot this information on
# separate plots, or specify a subset of frequencies to inspect. Note that the
# ``Figure`` and ``Axes`` objects can also be returned for any desired manual
# adjustments of the plots.

# %%

fig, axes = waveshape_sawtooths.results.plot(
    f1s=np.arange(15, 26),
    f2s=np.arange(15, 26),
    cbar_range_abs=[0, 1],
    cbar_range_real=[-1, 1],
    cbar_range_imag=[-1, 1],
    cbar_range_phase=[-1, 1],
)
fig, axes = waveshape_peaks_troughs.results.plot(
    f1s=np.arange(15, 26),
    f2s=np.arange(15, 26),
    cbar_range_abs=[0, 1],
    cbar_range_real=[-1, 1],
    cbar_range_imag=[-1, 1],
    cbar_range_phase=[-1, 1],
)

###############################################################################
# References
# -----------------------------------------------------------------------------
# .. footbibliography::

# %%
