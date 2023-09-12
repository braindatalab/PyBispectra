Motivation
==========

What is the bispectrum?
-----------------------
The bispectrum is a higher-order statistic, based on the Fourier transform of
the third order moment :footcite:`Nikias1987`. Two forms of computing the
bispectrum exist: the direct approach, in which the Fourier transform of the
data is computed and then used to compute the bispectum; or the indirect
approach, in which the third order moments of the data are computed first which
are then used to compute the bispectrum. The approach implemented in
PyBispectra is based on the direct method. The bispectrum, :math:`B`, has the
form

:math:`\large B_{kmn}(f_1,f_2)=<\vec{k}(f_1)\vec{m}(f_2)\vec{n}^*(f_2+f_1)>` ,

where :math:`kmn` is a combination of signals - whose data has been Fourier
transformed - and the angled brackets represent the average over epochs.

A normalised form of the bispectrum also exists, termed the bicoherence.
Several forms of normalisation exist, however a common form is the threenorm: a
univariate normalisation whereby the values of the bicoherence will be bound in
the range :math:`[0, 1]` in a manner that is independent of the coupling
properties within or between signals :footcite:`Shahbazi2014`. The threenorm,
:math:`N`, has the form

:math:`\large N_{kmn}(f_1,f_2)=(<|\vec{k}(f_1)|^3><|\vec{m}(f_2)|^3><|\vec{n}(f_2+f_1)|^3>)^{\frac{1}{3}}` .

The bicoherence, :math:`\mathcal{B}`, is then computed as

:math:`\large \mathcal{B}_{kmn}(f_1,f_2)=\Large\frac{B_{kmn}(f_1,f_2)}{N_{kmn}(f_1,f_2)}` .

There are several possible uses of the bispectrum and bicoherence for advanced
signal analyses, including for phase-amplitude coupling (a form of
cross-frequency coupling), waveform shape analysis, and time delay estimation.


Why analyse cross-frequency coupling, waveform shape, and time delay estimates?
-------------------------------------------------------------------------------
Cross-frequency coupling, waveform shape analysis, and time delay estimation
are useful for signal analysis across a range of disciplines.

Cross-frequency coupling methods allow us to analyse the interactions within
and across signals between a lower frequency, :math:`f_1`, and a higher
frequency, :math:`f_2`. Different forms of coupling exist, such as phase-phase
coupling, amplitude-amplitude coupling, and phase-amplitude coupling. E.g. in
phase-amplitude coupling, we examine the relationship between the phase of a
signal at :math:`f_1` and the amplitude of a signal at :math:`f_2`.
Cross-frequency interactions have been posited as fundamental aspects of
neuronal communication in the brain :footcite:`Canolty2010`, with alterations
in these relationships implicated in diseases such as Parkinson's
:footcite:`deHemptinne2013` and Alzheimer's :footcite:`Bazzigaluppi2018`.

Additionally, aspects of a waveform's shape can also contain information of
interest. For example, non-sinusoidal features of signals may reflect
particular forms of interneuronal communication :footcite:`Sherman2016`, and
have been shown to be correlated with symptoms of neurological diseases and
altered by their treatments :footcite:`Cole2017`.

On the other hand, time delay estimation methods allow us to analyse the
interactions across signals and frequencies by estimating the time delay,
:math:`\tau`, between signals. Estimating time delays is a crucial matter for
radar and sonar technologies :footcite:`Chen2004`, however it is also of
interest in other signal analysis contexts, such as again in neuroscience
where time delays can be used to infer features of the physical relationships
between interacting brain regions :footcite:`Silchenko2010`.


Why use the bispectrum for these analyses?
------------------------------------------
The bispectrum offers several advantages over other methods for analysing
phase-amplitude coupling, waveform shape, and time delay estimates.

For phase-amplitude coupling, common methods such as the modulation index can
be practically challenging, requiring a precise set of filters to be applied to
the data to extract the true underlying interactions (which are not readily
apparent) as well as being computationally expensive (due to the requirement of
Hilbert transforming the data) :footcite:`Zandvoort2021`. Furthermore, when
analysing coupling between separate signals, the modulation index can perform
poorly at distinguishing genuine across-site coupling from within-site coupling
in the presence of source mixing :footcite:`Chella2014`. The bispectrum,
however, overcomes these issues, being computationally cheaper, lacking the
need to precisely filter the data, and being able to detect genuine across-site
coupling with minimal influence from within-site interactions in the presence
of source mixing :footcite:`Chella2014,PellegriniInPrep`.

Additionally, analyses of waveform shape often rely on searching through the
time-series signal :footcite:`Cole2017`, a computationally expensive procedure
when handling long periods of high sampling-rate data. Furthermore, if
waveforms of particular frequencies are desired, the time-series must be
bandpass filtered, a process which can itself distort the shape of the
underlying waveform :footcite:`Bartz2019`. With the bispectrum, waveform shape
analysis can be performed in a computationally cheap, frequency-resolved manner
without the need to bandpass filter.

Finally, traditional forms of time delay estimation often rely on
cross-correlation. This method is perfectly adequate in noiseless situations or
those where the noise of the signals are uncorrelated with one another as well
as with the sources of interest :footcite:`Nikias1988,JurharInPrep`. This,
however, is often not a realistic assumption, leading to spurious time delay
estimates. In contrast, the bispectrum is able to suppress the contribution of
Gaussian noise sources to time delay estimates :footcite:`Nikias1988`, and
additional steps can be taken to minimise the effects of non-Gaussian noise
sources, such as those associated with volume conduction
:footcite:`JurharInPrep`.


What is available in PyBispectra?
---------------------------------
PyBispectra offers tools for computing phase-amplitude coupling, time delay
estimation, and wave shape feature analysis using the bispectrum and
bicoherence. Additional tools are included for computing phase-phase coupling,
amplitude-amplitude coupling, and Fourier coefficients, as well as tools for
performing generalised eigendecompositions and plotting results.

You can find the installation instructions :doc:`here <installation>`, as well
as examples of how the package can be used :doc:`here <examples>`.


References
----------
.. footbibliography::
