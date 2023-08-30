"""
================================
Compute phase-amplitude coupling
================================

This example demonstrates how phase-amplitude coupling (PAC) can be computed
with PyBispectra.
"""

# %%

import os
from pathlib import Path

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
# :math:`\large PAC(\vec{x}_{f_1},\vec{y}_{f_2})=B_{xyy}(f_1,f_2)=<\vec{x}(f_1)
# \vec{y}(f_2)\vec{y}^*(f_2+f_1)>`.
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
# where the resulting PAC results are normalised to the range :math:`[0, 1]` by
# the power of the corresponding frequencies. Furthermore, PAC can be
# antisymmetrised by subtracting the results from those found using the
# transposed bispectrum, :math:`B_{xyx}` :footcite:`Chella2014`. In the context
# of analysing PAC between two signals, antisymmetrisation allows you to
# correct for spurious estimates of coupling arising from interactions within
# the signals themselves in instances of source mixing, providing a more robust
# connectivty metric.

###############################################################################
# Loading data and computing Fourier coefficients
# -----------------------------------------------
# We will start by loading some simulated data containing coupling between the
# 10 Hz phase of one signal and the 60 Hz amplitude of another. We will then
# compute the Fourier coefficients of the data, which will be used to compute
# PAC.

# %%

data_dir = os.path.join(
    os.path.abspath(Path(os.getcwd().parent)), "_static", "data"
)

# load simulated data
data = np.load(os.path.join(data_dir, "sim_data_pac_bivariate.npy"))
sampling_freq = 200  # sampling frequency in Hz

# compute Fourier coeffs.
fft_coeffs, freqs = compute_fft(
    data=data,
    sampling_freq=sampling_freq,
    n_points=sampling_freq,
    verbose=False,
)

print(
    f"FFT coeffs.: [{fft_coeffs.shape[0]} epochs x {fft_coeffs.shape[1]} "
    f"channels x {fft_coeffs.shape[2]} frequencies]\nFreq. range: {freqs[0]} "
    f"- {freqs[-1]} Hz"
)

###############################################################################
# As you can see, we have FFT coefficients for 2 channels across 30 epochs,
# with 101 frequencies ranging from 0 to 100 Hz with a frequency resolution of
# 1 Hz. We will use these coefficients to compute PAC.
#
# Computing PAC
# -------------
# To compute PAC, we start by initialising the :class:`~pybispectra.cfc.PAC`
# class object with the FFT coefficients and the frequency information. To
# compute PAC, we call the :meth:`~pybispectra.cfc.PAC.compute` method. By
# default, PAC is computed between all channel and frequency combinations,
# however we can also specify particular combinations of interest.
#
# Here, we specify the :attr:`~pybispectra.cfc.PAC.indices` to compute PAC on.
# :attr:`~pybispectra.cfc.PAC.indices` is expected to be a tuple containing two
# NumPy arrays for the indices of the seed and target channels, respectively.
# The indices specified below mean that PAC will only be computed across
# frequencies between the channels (i.e. 0 -> 1). By leaving the frequency
# arguments :attr:`~pybispectra.cfc.PAC.f1s` and
# :attr:`~pybispectra.cfc.PAC.f2s` blank, we will look at all possible
# frequency combinations.

# %%

pac = PAC(
    data=fft_coeffs, freqs=freqs, sampling_freq=sampling_freq, verbose=False
)  # initialise object
pac.compute(indices=([0], [1]))  # compute PAC

pac_results = pac.results.get_results()  # return results as array

print(
    f"PAC results: [{pac_results.shape[0]} connections x "
    f"{pac_results.shape[1]} f1s x {pac_results.shape[2]} f2s]"
)

###############################################################################
# We can see that PAC has been computed for one connection (0 -> 1), and all
# possible frequency combinations, averaged across our 30 epochs. Whilst there
# are > 10,000 such frequency combinations in our [101 x 101] matrix, PAC for
# those entries where :math:`f_1` would be higher than :math:`f_2`, as well as
# where :math:`f_2 + f_1` exceeds the frequency bounds of our data, cannot be
# computed. In such cases, the values are ``numpy.nan``.

###############################################################################
# Plotting PAC
# ------------
# Let us now inspect the results. Here, we specify a subset of frequencies to
# inspect around the simulated interaction. If we wished, we could also plot
# all frequencies. Note that the :class:`~matplotlib.figure.Figure` and
# :class:`~matplotlib.axes.Axes` objects can also be returned for any desired
# manual adjustments of the plots. In this simulated data example, we can see
# that the bispectrum indeed identifies the occurence of 10-60 Hz PAC between
# our seed and target channels.

# %%

fig, axes = pac.results.plot(f1s=np.arange(5, 16), f2s=np.arange(55, 66))

###############################################################################
# Antisymmetrisation for across-signal PAC
# ----------------------------------------
# In this simulated data example, interactions are only present between the
# signals, and not within the signals themselves. This, however, is not always
# the case, and estimates of across-site PAC can be corrupted by coupling
# interactions within each signal in the presence of source mixing. To combat
# this, we can employ antisymmetrisation :footcite:`Chella2014`. The example
# below shows some such simulated data consisting of two independent sources,
# with 10-60 Hz PAC within each source (top two plots), as well as a mixing of
# the underlying sources to produce 10-60 Hz PAC between the two signals
# (bottom left plot). When appyling antisymmetrisation, however, we see that
# the spurious across-signal PAC arising from the source mixing is suppressed
# (bottom right plot). Antisymmetrisation is therefore a useful technique to
# differentiate genuine across-site coupling from spurious coupling arising
# from the within-site interactions of source-mixed signals.

# %%

# load real data
data = np.load(os.path.join(data_dir, "sim_data_pac_univariate.npy"))
sampling_freq = 200

# compute Fourier coeffs.
fft_coeffs, freqs = compute_fft(
    data=data,
    sampling_freq=sampling_freq,
    n_points=sampling_freq,
    verbose=False,
)

# compute PAC
pac = PAC(
    data=fft_coeffs, freqs=freqs, sampling_freq=sampling_freq, verbose=False
)
pac.compute(indices=([0, 1, 0], [0, 1, 1]), symmetrise=["none", "antisym"])
pac_standard, pac_antisym = pac.results

vmin = np.min(
    (
        np.nanmin(pac_standard.get_results()),
        np.nanmin(pac_antisym.get_results()),
    )
)
vmax = np.max(
    (
        np.nanmax(pac_standard.get_results()),
        np.nanmax(pac_antisym.get_results()),
    )
)

# plot unsymmetrised PAC within & between signals
fig_standard, axes_standard = pac_standard.plot(
    f1s=np.arange(5, 16), f2s=np.arange(55, 66), cbar_range=[vmin, vmax]
)

# plot antisymmetrised PAC between signals
fig_antisym, axes_antisym = pac_antisym.plot(
    nodes=[2],
    f1s=np.arange(5, 16),
    f2s=np.arange(55, 66),
    cbar_range=[vmin, vmax],
)

###############################################################################
# References
# -----------------------------------------------------------------------------
# .. footbibliography::

# %%
