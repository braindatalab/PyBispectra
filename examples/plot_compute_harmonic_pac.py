"""
==================================================================
Distinguishing harmonic from non-harmonic phase-amplitude coupling
==================================================================

This example demonstrates how the tools of PyBispectra (phase-phase coupling
(PPC), amplitude-amplitude coupling (AAC) and wave shape features) can be used
to identify phase-amplitude coupling (PAC) resulting from frequency harmonics,
as opposed to an interaction between distinct oscillations.
"""

# %%

import numpy as np

from pybispectra import compute_fft, PAC, PPC, WaveShape, SpatioSpectralFilter

###############################################################################
# Background
# ----------
# Typically, PAC is interpreted as an interaction between distinct oscillatory
# signals, whereby the phase of a slower oscillation modulates the amplitude of
# a faster oscillation, so-called non-harmonic PAC. However, PAC can also
# occur as a result of higher-frequency harmonics of a lower frequency
# oscillation of interest, termed harmonic PAC, which may be linked to how
# non-sinusoidal this lower frequency oscillation is :footcite:`Giehl2021`.
#
# Crucially, the mechanisms by which harmonic and non-harmonic PAC occur are
# thought to differ, which can have serious implications for how one interpets
# the results of any PAC analysis. Therefore, it is important to determine
# whether PAC is of the harmonic or non-harmonic variety. This can be done by
# comparing PAC with PPC, AAC, as well as wave shape features
# :footcite:`Giehl2021`.

###############################################################################
# Computing the various metrics
# -----------------------------
# We start by generating some data, computing its Fourier coefficients, and
# then computing PAC, PPC, AAC, and wave shape features (see the respective
# examples for detailed information on how to do this).
