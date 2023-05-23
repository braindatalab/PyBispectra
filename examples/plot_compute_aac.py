"""
====================================
Compute amplitude-amplitude coupling
====================================

This example demonstrates how amplitude-amplitude coupling (AAC) can be
computed with PyBispectra.
"""

# %%

import numpy as np

from pybispectra import compute_fft, AAC

###############################################################################
# Background
# ----------
# AAC quantifies the relationship between the amplitudes of a lower frequency
# :math:`f_1` and a higher frequency :math:`f_2` within a single signal, or
# across different signals. This is computed as the Pearson correlation
# coefficient between the power of the signals at :math:`f_1` and :math:`f_2`.

###############################################################################
# Generating data and computing Fourier coefficients
# --------------------------------------------------
# We will start by generating some data that we can compute AAC on, then
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
# with 101 frequencies ranging from 0 to 50 Hz with a frequency resolution of
# 0.5 Hz. We will use these coefficients to compute AAC.
#
# Computing AAC
# -------------
# To compute AAC, we start by initialising the :class:`AAC` class object with
# the FFT coefficients and the frequency information. To compute AAC, we call
# the :meth:`compute` method. By default, AAC is computed between all channel
# and frequency combinations, however we can also specify particular
# combinations of interest.
#
# Here, we specify the :attr:`indices` to compute AAC on. :attr:`indices` is
# expected to be a tuple containing two NumPy arrays for the indices of the
# seed and target channels, respectively. The indices specified below mean that
# AAC will only be computed across frequencies within each channel (i.e.
# 0 -> 0; and 1 -> 1). By leaving the frequency arguments :attr:`f1s` and
# :attr:`f2s` blank, we will look at all possible frequency combinations.

# %%

aac = AAC(data=fft, freqs=freqs, sampling_freq=sfreq)  # initialise object
aac.compute(indices=(np.array([0]), np.array([1])))  # compute AAC

aac_results = aac.results.get_results()  # return results as array

print(
    f"PPC results: [{aac_results.shape[0]} connections x "
    f"{aac_results.shape[1]} f1s x {aac_results.shape[2]} f2s]"
)

###############################################################################
# We can see that AAC has been computed for 2 connections (0 -> 0; and 1 -> 1),
# and all possible frequency combinations, averaged across our 30 epochs.
# Whilst there are > 10,000 such frequency combinations in our [101 x 101]
# matrices, AAC for the lower triangular matrices are naturally mirrors of the
# upper triangular matrices. Accordingly, the values for these redundant
# entries are left as ``numpy.nan`` (see the plotted results below for a visual
# demonstration of this).

###############################################################################
# Plotting AAC
# ------------
# Let us now inspect the results. For this, we will plot the results for both
# connections on the same plot. If we wished, we could plot this information on
# separate plots, or specify a subset of frequencies to inspect. Note that the
# ``Figure`` and ``Axes`` objects can also be returned for any desired manual
# adjustments of the plots.

# %%

fig, axes = aac.results.plot(major_tick_intervals=5.0)

# %%
