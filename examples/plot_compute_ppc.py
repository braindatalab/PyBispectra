"""
============================
Compute phase-phase coupling
============================

This example demonstrates how phase-phase coupling (PPC) can be computed with
PyBispectra.
"""

# %%

import os
from pathlib import Path

import numpy as np

from pybispectra import compute_fft, PPC

###############################################################################
# Background
# ----------
# PPC quantifies the relationship between the phases of a lower frequency
# :math:`f_1` and a higher frequency :math:`f_2` within a single signal, or
# across different signals.
#
# The method available in PyBispectra can be thought of as a measure of
# coherence between frequencies :footcite:`Giehl2021` (note that it is not
# based on the bispectrum):
#
# :math:`\large PPC(\vec{x}_{f_1},\vec{y}_{f_2})=\LARGE \frac{|\langle
# \vec{a}_x(f_1)\vec{a}_y(f_2) e^{i(\vec{\varphi}_x(f_1)\frac{f_2}{f_1}-
# \vec{\varphi}_y(f_2))} \rangle|}{\langle \vec{a}_x(f_1)\vec{a}_y(f_2)
# \rangle}`,
#
# where :math:`\vec{a}(f)` and :math:`\vec{\varphi}(f)` are the amplitude and
# phase of a signal at a given frequency, respectively, and the angled brackets
# represent the average over epochs. The phase of :math:`f_1` is accelerated to
# match that of :math:`f_2` by scaling the phase by a factor of
# :math:`\frac{f_2}{f_1}`. PPC values for this measure lie in the range
# :math:`[0, 1]`, with 0 representing a random phase relationship, and 1
# representing perfect phase coupling.

###############################################################################
# Generating data and computing Fourier coefficients
# --------------------------------------------------
# We will start by loading some data that we can compute PPC on, then compute
# the Fourier coefficients of the data.

# %%

data_dir = os.path.join(
    os.path.abspath(Path(os.getcwd()).parent), "_static", "data"
)

# generate data
data = np.load(os.path.join(data_dir, "sim_data_ppc.npy"))
sampling_freq = 500  # Hz

# compute Fourier coeffs.
fft_coeffs, freqs = compute_fft(
    data=data,
    sampling_freq=sampling_freq,
    n_points=sampling_freq,
    verbose=False,
)

print(
    f"FFT coeffs.: [{fft_coeffs.shape[0]} epochs x {fft_coeffs.shape[1]} "
    f"channels x {fft_coeffs.shape[2]} frequencies]\n"
    f"Freq. range: {freqs[0]} - {freqs[-1]} Hz"
)

###############################################################################
# As you can see, we have FFT coefficients for 2 channels across 30 epochs,
# with 101 frequencies ranging from 0 to 50 Hz with a frequency resolution of
# 0.5 Hz. We will use these coefficients to compute PPC.
#
# Computing PPC
# -------------
# To compute PPC, we start by initialising the :class:`~pybispectra.cfc.PPC`
# class object with the FFT coefficients and the frequency information. To
# compute PPC, we call the :meth:`~pybispectra.cfc.PPC.compute` method. By
# default, PPC is computed between all channel and frequency combinations,
# however we can also specify particular combinations of interest.
#
# Here, we specify the :attr:`~pybispectra.cfc.PPC.indices` to compute PPC on.
# :attr:`~pybispectra.cfc.PPC.indices` is expected to be a tuple containing two
# NumPy arrays for the indices of the seed and target channels, respectively.
# The indices specified below mean that PPC will only be computed across
# frequencies between channels (i.e. 0 -> 1). By leaving the frequency
# arguments :attr:`~pybispectra.cfc.PPC.f1s` and
# :attr:`~pybispectra.cfc.PPC.f2s` blank, we will look at all possible
# frequency combinations.

# %%

ppc = PPC(
    data=fft_coeffs, freqs=freqs, sampling_freq=sampling_freq, verbose=False
)  # initialise object
ppc.compute(indices=([0], [1]))  # compute PPC
ppc_results = ppc.results.get_results()  # return results as array

print(
    f"PPC results: [{ppc_results.shape[0]} connection(s) x "
    f"{ppc_results.shape[1]} f1 x {ppc_results.shape[2]} f2]"
)

###############################################################################
# We can see that PPC has been computed for one connection (0 -> 1), and all
# possible frequency combinations, averaged across our 30 epochs. Whilst there
# are 10,000 such frequency combinations in our [100 x 100] matrices, PPC for
# those entries where :math:`f1` would be higher than :math:`f2` is not
# computed, in which case the values are ``numpy.nan``.

###############################################################################
# Plotting PPC
# ------------
# Let us now inspect the results. For this, we will plot the results for all
# frequencies, although we could specify a subset of frequencies to inspect.

# %%

ppc.results.plot(f1s=np.arange(5, 16), f2s=np.arange(55, 66))

###############################################################################
# As you can see, values for the lower right triangle of each plot are missing,
# corresponding to the frequency combinations where :math:`f_1` is greater than
# :math:`f_2`, and hence where PPC is not computed. Note that the ``Figure``
# and ``Axes`` objects can also be returned for any desired manual adjustments
# of the plots.

###############################################################################
# References
# -----------------------------------------------------------------------------
# .. footbibliography::

# %%
