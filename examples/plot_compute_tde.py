"""
============================
Compute time delay estimates
============================

This example demonstrates how time delay estimation (TDE) can be computed with
PyBispectra.
"""

# %%

import os
from pathlib import Path

import numpy as np

from pybispectra import compute_fft, TDE

###############################################################################
# Background
# ----------
# A common feature of interest in signal analyses is determining the flow of
# information between two signals, in terms of both the direction and the
# particular time lag between them, known as TDE.
#
# The method available in PyBispectra is based on the bispectrum, :math:`B`,
# which has the general form:
#
# :math:`\large B_{kmn}(f_1,f_2)=<\vec{k}(f_1)\vec{m}(f_2)\vec{n}^*(f_2+f_1)>`,
#
# where :math:`kmn` is a combination of channels :math:`\vec{x}` and
# :math:`\vec{y}`, and the angled brackets represent the averaged value over
# epochs. When computing TDE, information from :math:`\vec{n}` is taken not
# only from the positive frequencies, but also the negative frequencies. Four
# methods exist for computing TDE based on the bispectrum
# :footcite:`Nikias1988`. The fundamental equation is as follows:
#
# :math:`\large TDE_{xy}(\tau)=\int_{-\pi}^{+\pi}\int_{-\pi}^{+\pi}I(
# \vec{x}_{f_1},\vec{y}_{f_2})e^{-if_1\tau}df_1df_2`,
#
# where :math:`I` varies depending on the method, and :math:`\tau` is a given
# time delay. Phase information of the signals is extracted from the bispectrum
# in two variants used by the different methods:
#
# :math:`\large \phi(\vec{x}_{f_1},\vec{y}_{f_2})=\varphi_{B_{xyx}} (f_1,f_2)-
# \varphi_{B_{xxx}}(f_1,f_2)`
#
# :math:`\large \phi'(\vec{x}_{f_1},\vec{y}_{f_2})=\varphi_{B_{xyx}} (f_1,f_2)-
# \frac{1}{2}(\varphi_{B_{xxx}}(f_1, f_2) + \varphi_{B_{yyy}} (f_1,f_2))`
#
# **Method I**:
# :math:`\large I(\vec{x}_{f_1},\vec{y}_{f_2})=e^{i\phi(\vec{x}_{f_1},
# \vec{y}_{f_2})}`
#
# **Method II**:
# :math:`\large I(\vec{x}_{f_1},\vec{y}_{f_2})=e^{i\phi'(\vec{x}_{f_1},
# \vec{y}_{f_2})}`
#
# **Method III**:
# :math:`\large I(\vec{x}_{f_1},\vec{y}_{f_2})=\Large \frac{B_{xyx}
# (f_1,f_2)}{B_{xxx}(f_1,f_2)}`
#
# **Method IV**:
# :math:`\large I(\vec{x}_{f_1},\vec{y}_{f_2})=\Large \frac{|B_{xyx}
# (f_1,f_2)|e^{i\phi'(\vec{x}_{f_1},\vec{y}_{f_2})}}{\sqrt{|B_{xxx}
# (f_1,f_2)||B_{yyy}(f_1,f_2)|}}`
#
# where :math:`\varphi_{B}` is the phase of the bispectrum. All four methods
# aim to capture the phase difference between :math:`\vec{x}` and
# :math:`\vec{y}`. Method I involves the extraction of phase spectrum
# periodicity and monotomy, with method III involving an additional amplitude
# weighting from the bispectrum of :math:`\vec{x}`. Method II instead relies on
# a combination of phase spectra of the different frequency components, with
# method IV containing an additional amplitude weighting from the bispectra of
# :math:`\vec{x}` and :math:`\vec{y}`. No single method is superior to another.
#
# As a result of volume conduction artefacts (i.e. a common underlying signal
# that propagates instantaneously to :math:`\vec{x}` and :math:`\vec{y}`), time
# delay estimates can become contaminated, resulting in spurious estimates of
# :math:`\tau=0`. Thankfully, antisymmetrisation of the bispectrum can be used
# to address these mixing artefacts :footcite:`Chella2014`, which is
# implemented here as the replacement of :math:`B_{xyx}` with :math:`(B_{xxy} -
# B_{yxx})` in the above equations :footcite:`JurharInPrep`.
#
# As a final note, if TDE for only certain frequencies is of interest, the
# signals can be bandpass filtered prior to computing the bispectrum.

###############################################################################
# Loading data and computing Fourier coefficients
# -----------------------------------------------
# We will start by loading some simulated data containing a time delay of 250
# ms between two signals, where :math:`\vec{y}` is a delayed version of
# :math:`\vec{x}`. We will then compute the Fourier coefficients of the data,
# which will be used to compute TDE. Since TDE requires information from both
# negative and positive frequencies, we set the ``return_neg_freqs`` parameter
# to ``True``. Furthermore, we specify ``n_points`` to be twice the number of
# time points in the data, plus one, to ensure that the time delay estimates
# correspond to the sampling frequency of the data (accounting for time point
# zero as well as the fact that the estimates are returned for both time delay
# directions, i.e. where :math:`\vec{x}` drives :math:`\vec{y}` and
# :math:`\vec{y}` drives :math:`\vec{x}`). By altering the number of points
# used to compute the Fourier coefficients, the temporal resolution of the TDE
# results can be adjusted. E.g. a higher number of points increases the
# temporal resolution at the cost of computational cost.
#
# In this example, our data consists of 30 epochs of 200 timepoints each, which
# with a 200 Hz sampling frequency corresponds to 1 second of data per epoch
# (one timepoint every 5 ms). By specifying ``n_points=2 * n_times + 1``, we
# will obtain delay estimates at a resolution of 5 ms.

# %%

data_dir = os.path.join("..", "_static", "data")

# load simulated data
data = np.load(os.path.join(data_dir, "sim_data_tde_independent_noise.npy"))
sampling_freq = 200  # sampling frequency in Hz
n_times = data.shape[2]  # number of timepoints in the data

# compute Fourier coeffs.
fft_coeffs, freqs = compute_fft(
    data=data,
    sampling_freq=sampling_freq,
    n_points=2 * n_times + 1,
    window="hamming",
    return_neg_freqs=True,
    verbose=False,
)

print(
    f"FFT coeffs.: [{fft_coeffs.shape[0]} epochs x {fft_coeffs.shape[1]} "
    f"channels x {fft_coeffs.shape[2]} frequencies]\nFreq. range: "
    f"{freqs[freqs.argmin()] :.0f} - {freqs[freqs.argmax()] :.0f} Hz"
)

###############################################################################
# Computing TDE
# -------------
# To compute TDE, we start by initialising the :class:`~pybispectra.tde.TDE`
# class object with the FFT coefficients and the frequency information. To
# compute TDE, we call the :meth:`~pybispectra.tde.TDE.compute` method. To keep
# things simple, we will focus on TDE using method I. To demonstrate that TDE
# can show the directionality of information flow as well as the particular
# time lag, we will compute TDE from signals 0 -> 1 (the genuine direction of
# information flow where the time delay should have a positive value) and from
# signals 1 -> 0 (the reverse direction of information flow where the time
# delay should have a negative value).

# %%

tde = TDE(
    data=fft_coeffs, freqs=freqs, sampling_freq=sampling_freq, verbose=False
)  # initialise object
tde.compute(indices=([0, 1], [1, 0]), method=1)  # compute TDE
tde_times = tde.results.times

tde_results = tde.results.get_results()  # return results as array

print(
    f"PAC results: [{tde_results.shape[0]} connections x "
    f"{tde_results.shape[1]} times]"
)

###############################################################################
# We can see that TDE has been computed for two connections (0 -> 1 and 1 ->
# 0), and 401 timepoints (twice that of the original data plus one, preserving
# the sampling frequency of the data - i.e. with one estimate every 5 ms - and
# including the zero time), averaged across our 30 epochs.

###############################################################################
# Plotting TDE
# ------------
# Let us now inspect the results. Note that the
# :class:`~matplotlib.figure.Figure` and :class:`~matplotlib.axes.Axes` objects
# can be returned for any desired manual adjustments of the plots. When
# handling TDE results, we take the time at which the strength of the estimate
# is maximal as our :math:`\tau`. Doing so, we indeed see that the time delay
# is identified as 250 ms. Furthermore, comparing the two connections, we see
# that the direction of information flow is also correctly identified, with the
# result for connection 0 -> 1 being positive and the result for connection 1
# -> 0 being negative (i.e. information flow from signal 0 to signal 1). Here,
# we manually find :math:`\tau` based on the maximal value of the TDE results,
# however this information is also precomputed and can be accessed via the
# :attr:`~pybispectra.utils.ResultsTDE.tau` attribute.
#
# Taking the time at which the estimate is maximal as our :math:`\tau` is the
# approach to use when estimating time delays. For interest, however, we can
# also plot the full time course of the TDE results. In this low noise example,
# we see that there is a clear peak in time delay estimates at 250 ms.
# Depending on the nature and degree of noise in the data, the time delay
# spectra may be less clear, and you may find advantages using the other
# TDE method variants.

# %%

print(
    "The estimated time delay between signals 0 and 1 is "
    f"{tde_times[tde_results[0].argmax()]:.0f} ms.\n"
    "The estimated time delay between signals 1 and 0 is "
    f"{tde_times[tde_results[1].argmax()]:.0f} ms."
)

for con_i in range(tde_results.shape[0]):
    assert tde_times[tde_results[con_i].argmax()] == tde.results.tau[con_i]

fig, axes = tde.results.plot()

###############################################################################
# Handling artefacts from volume conduction
# -----------------------------------------
# In the example above, we looked at simulated data from two signals with
# independent noise sources, giving a clean TDE result at the true delay. In
# real data, however, sources of noise in the data are often correlated across
# signals, such as due to volume conduction, resulting in a bias of TDE methods
# towards zero time delay. To mitigate such bias, we can employ
# antisymmetrisation of the bispectrum :footcite:`JurharInPrep`. To demonstrate
# this, we will now look at simulated data (still with a 250 ms delay) with the
# addition of a common underlying noise source between the signals.
#
# As you can see, the TDE result without antisymmetrisation consists of two
# distinct peaks: a larger one at time zero; and a smaller one at the genuine
# time delay (250 ms). As the estimate at time zero is largest, :math:`\tau` is
# therefore incorrectly determined to be 0 ms. In contrast, antisymmetrisation
# suppresses the spurious peak at time zero, leaving only a clear peak at the
# genuine time delay and the correct estimation of :math:`\tau`. Accordingly,
# in instances where there is a risk of correlated noise sources between the
# signals (e.g. with volume conduction), applying antisymmetrisation when
# estimating time delays is recommended.

# %%

# load simulated data
data = np.load(os.path.join(data_dir, "sim_data_tde_correlated_noise.npy"))
sampling_freq = 200  # sampling frequency in Hz
n_times = data.shape[2]  # number of timepoints in the data

# compute Fourier coeffs.
fft_coeffs, freqs = compute_fft(
    data=data,
    sampling_freq=sampling_freq,
    n_points=2 * n_times + 1,
    window="hamming",
    return_neg_freqs=True,
    verbose=False,
)

# compute TDE
tde = TDE(
    data=fft_coeffs, freqs=freqs, sampling_freq=sampling_freq, verbose=False
)
tde.compute(indices=([0], [1]), symmetrise=["none", "antisym"], method=1)
tde_standard, tde_antisym = tde.results

print(
    "The estimated time delay without antisymmetrisation is "
    f"{tde_standard.tau[0]:.0f} ms.\n"
    "The estimated time delay with antisymmetrisation is "
    f"{tde_antisym.tau[0]:.0f} ms."
)

# plot results
tde_standard.plot()
tde_antisym.plot()

# %%
