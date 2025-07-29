Motivation
==========

What is the bispectrum?
-----------------------
The bispectrum is a higher-order statistic, based on the Fourier transform of
the third order moment :footcite:`Nikias1987`. Two forms of computing the
bispectrum exist: the direct approach, in which the Fourier coefficients of the
data are computed, which in turn are used to compute the bispectum; or the
indirect approach, in which the third order moments of the data are computed
first before the Fourier transform is taken. PyBispectra uses the direct
method. The bispectrum, :math:`\textbf{B}`, has the form

:math:`\textbf{B}_{kmn}(f_1,f_2)=<\textbf{k}(f_1)\textbf{m}(f_2)\textbf{n}^*(f_2+f_1)>` ,

where :math:`kmn` is a combination of signals with Fourier coefficients
:math:`\textbf{k}`, :math:`\textbf{m}`, and :math:`\textbf{n}`, respectively;
:math:`f_1` and :math:`f_2` correspond to a lower and higher frequency,
respectively; and :math:`<>` represents the average value over epochs.

A normalised form of the bispectrum also exists, termed bicoherence. Several
forms of normalisation exist, however a common form is the threenorm: a
univariate normalisation whereby the values of the bicoherence will be bound in
the range :math:`[-1, 1]` in a manner that is independent of the coupling
properties within or between signals :footcite:`Shahbazi2014`. The threenorm,
:math:`\textbf{N}`, has the form

:math:`\textbf{N}_{kmn}(f_1,f_2)=(<|\textbf{k}(f_1)|^3><|\textbf{m}(f_2)|^3><|\textbf{n}(f_2+f_1)|^3>)^{\frac{1}{3}}` .

The bicoherence, :math:`\boldsymbol{\mathcal{B}}`, is then computed as

:math:`\boldsymbol{\mathcal{B}}_{kmn}(f_1,f_2)=\Large\frac{\textbf{B}_{kmn}(f_1,f_2)}{\textbf{N}_{kmn}(f_1,f_2)}` .

There are several possible uses of the bispectrum and bicoherence for signal
analyses, including for phase-amplitude coupling (a form of cross-frequency
coupling), the analysis of non-sinusoidal waveform features, and time delay
estimation.


Why analyse cross-frequency coupling, waveshape, and time delays?
-----------------------------------------------------------------
Cross-frequency coupling, waveshape analysis, and time delay estimation are
relevant in a range of disciplines.

Cross-frequency coupling methods allow us to analyse interactions within and
across signals between a lower frequency, :math:`f_1`, and a higher frequency,
:math:`f_2`. Different forms of coupling exist, such as phase-phase coupling,
amplitude-amplitude coupling, and phase-amplitude coupling. In phase-amplitude
coupling, we examine the relationship between the phase of a signal at
:math:`f_1` and the amplitude of a signal at :math:`f_2`. Cross-frequency
interactions have been posited as fundamental aspects of neuronal communication
in the brain :footcite:`Canolty2010`, with alterations in these relationships
implicated in diseases such as Parkinson's :footcite:`deHemptinne2013` and
Alzheimer's :footcite:`Bazzigaluppi2018`.

Additionally, a signal's shape can contain information of interest. For
example, non-sinusoidal features of signals may reflect particular forms of
interneuronal communication :footcite:`Sherman2016`, and have been shown to be 
correlated with symptoms of neurological diseases and altered by their
treatments :footcite:`Cole2017`.

Finally, time delays, :math:`\tau`, between signals can also provide useful
insights into systems. Such estimates are crucial for radar and sonar
technologies :footcite:`Chen2004`, but also in neuroscience, where time delays
can be used to infer features of the physical relationships between interacting
brain regions :footcite:`Silchenko2010`.

Ultimately, the bispectrum is a useful technique for the analysis of
electrophysiological signals. This includes (non-)invasive neural data such as
EEG, MEG, ECoG, and LFP, but also non-neural data like EMG and ECG.


Why use the bispectrum for these analyses?
------------------------------------------
The bispectrum offers several advantages over other methods for analysing
phase-amplitude coupling, waveform shape, and time delay estimates.

For phase-amplitude coupling, common methods such as the modulation index can
be practically challenging, requiring a precise set of filters to be applied to
the data to extract the true underlying interactions (which are not readily
apparent) as well as being computationally expensive (due to the requirement of
Hilbert transforming the data) :footcite:`Zandvoort2021`. Furthermore, when
analysing coupling between separate signals, the modulation index performs
poorly at distinguishing genuine across-site coupling from within-site coupling
in the presence of source mixing :footcite:`Chella2014`. The bispectrum
overcomes these issues, being computationally cheaper, lacking the
need to precisely filter the data :footcite:`Zandvoort2021`, and being more
robust to spurious across-site coupling estimates
:footcite:`PellegriniPreprint`.

Additionally, analyses of waveshape often rely on searching through the
time-series signal :footcite:`Cole2017`, a computationally expensive procedure
when handling long periods of high sampling-rate data. Furthermore, if
information at particular frequencies is desired, the time-series must be
bandpass filtered, distorting the shape of non-sinusoidal aspects of the
underlying signal :footcite:`Bartz2019`. With the bispectrum, non-sinudoisal
waveshape information can be extracted in a computationally cheap,
frequency-resolved manner, without the need to bandpass filter.

Finally, traditional forms of time delay estimation often rely on
cross-correlation. This method is perfectly adequate in noiseless situations or
those where the noise of the signals are uncorrelated with one another as well
as with the sources of interest :footcite:`Nikias1988,JurharPreprint`. This,
however, is often not the case in many real-world contexts, leading to spurious
time delay estimates. In contrast, the bispectrum is able to suppress the
contribution of Gaussian noise sources to time delay estimates
:footcite:`Nikias1988`, and additional steps can be taken to minimise the
effects of non-Gaussian noise sources, such as those associated with volume
conduction :footcite:`JurharPreprint`.


What is available in PyBispectra?
---------------------------------
PyBispectra offers tools for computing phase-amplitude coupling, time delay
estimation, and waveshape feature analysis using the bispectrum and
bicoherence. Additional tools are included for computing phase-phase coupling,
amplitude-amplitude coupling, Fourier coefficients, time-frequency
representations of data, spatio-spectral filters, as well as plotting results.

You can find the installation instructions :doc:`here <installation>`, as well
as examples of how the package can be used :doc:`here <examples>`.


References
----------
.. footbibliography::
