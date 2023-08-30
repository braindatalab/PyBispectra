"""
====================================
Compute amplitude-amplitude coupling
====================================

This example demonstrates how amplitude-amplitude coupling (AAC) can be
computed with PyBispectra.
"""

# %%

import os
from pathlib import Path

import numpy as np

from pybispectra import compute_tfr, AAC

###############################################################################
# Background
# ----------
# AAC quantifies the relationship between the amplitudes of a lower frequency
# :math:`f_1` and a higher frequency :math:`f_2` within a single signal, or
# across different signals. This is computed as the Pearson correlation
# coefficient between the power of the time-frequency representation (TFR) of
# the signals at :math:`f_1` and :math:`f_2` across time, averaged over epochs
# :footcite:`Giehl2021` (i.e. it is not based in the bispectrum).

###############################################################################
# Generating data and computing Fourier coefficients
# --------------------------------------------------
# We will start by loading some data that we can compute AAC on, then compute
# the amplitude TFR of the data (using Morlet wavelets in this example).

# %%

data_dir = os.path.abspath(os.path.join("..", "_static", "data"))

# load example data
data = np.load(os.path.join(data_dir, "sim_data_aac.npy"))
sampling_freq = 200  # Hz
freqs = np.arange(5, 101, 1)

# compute amplitude TFR
tfr, freqs = compute_tfr(
    data=data,
    sampling_freq=sampling_freq,
    freqs=freqs,
    tfr_mode="multitaper",
    n_cycles=7,
    verbose=False,
)

print(
    f"TFR of data: [{tfr.shape[0]} epochs x {tfr.shape[1]} channels x "
    f"{tfr.shape[2]} frequencies x {tfr.shape[3]} times]\nFreq. range: "
    f"{freqs[0]} - {freqs[-1]} Hz"
)

###############################################################################
# As you can see, we have the amplitude TFR for 2 channels for 30 epochs, with
# frequencies ranging from 1 to 100 Hz (1 Hz resolution), and 400 timepoints
# per epoch. The amplitude TFR of the data will be used to compute AAC.
#
# Computing AAC
# -------------
# To compute AAC, we start by initialising the :class:`~pybispectra.cfc.AAC`
# class object with the amplitude TFR and the frequency information. To compute
# AAC, we call the :meth:`~pybispectra.cfc.AAC.compute` method. By default, AAC
# is computed between all channel and frequency combinations, however we can
# also specify particular combinations of interest.
#
# Here, we specify the :attr:`~pybispectra.cfc.AAC.indices` to compute AAC on.
# :attr:`~pybispectra.cfc.AAC.indices` is expected to be a tuple containing two
# NumPy arrays for the indices of the seed and target channels, respectively.
# The indices specified below mean that AAC will only be computed across
# frequencies between each channel (i.e. 0 -> 1). By leaving the frequency
# arguments :attr:`~pybispectra.cfc.AAC.f1s` and
# :attr:`~pybispectra.cfc.AAC.f2s` blank, we will look at all possible
# frequency combinations.

# %%

aac = AAC(
    data=tfr, freqs=freqs, sampling_freq=sampling_freq, verbose=False
)  # initialise object
aac.compute(indices=([1], [0]))  # compute AAC
aac_results = aac.results.get_results()  # return results as array

print(
    f"AAC results: [{aac_results.shape[0]} connection(s) x "
    f"{aac_results.shape[1]} f1s x {aac_results.shape[2]} f2s]"
)

###############################################################################
# We can see that AAC has been computed for 1 connections (0 -> 1), and all
# possible frequency combinations, averaged across our 30 epochs. Whilst there
# are 10,000 such frequency combinations in our [100 x 100] matrix, AAC for the
# lower triangular matrix is naturally a mirror of the upper triangular matrix.
# Accordingly, the values for these redundant entries are left as ``numpy.nan``
# (see the plotted results below for a visual demonstration of this).

###############################################################################
# Plotting AAC
# ------------
# Let us now inspect the results. For this, we will plot the results for all
# frequencies, although we could specify a subset of frequencies to inspect.

# %%

fig, axes = aac.results.plot()  # f1s=np.arange(5, 16), f2s=np.arange(55, 66))

###############################################################################
# As you can see, values for the lower right triangle of each plot are missing,
# corresponding to the frequency combinations where :math:`f_1` is greater than
# :math:`f_2`, and hence where PPC is not computed. Note that the ``Figure``
# and ``Axes`` objects can also be returned for any desired manual adjustments
# of the plots.

# %%
