---
title: 'PyBispectra: A toolbox for advanced electrophysiological signal processing using the bispectrum'
tags:
  - Python
  - neuroscience
  - signal processing
  - bispectrum
authors:
  - name: Thomas S. Binns
    orcid: 0000-0003-0657-0891
    affiliation: "1, 2, 3"
    corresponding: true
  - name: Franziska Pellegrini
    orcid: 0000-0001-9769-1597
    affiliation: "3, 4"
  - name: Tin Jurhar
    orcid: 0000-0002-8804-2349
    affiliation: "5, 6"
  - name: Tien D. Nguyen
    orcid: 0009-0008-9867-0964
    affiliation: "3, 4"
  - name: Richard M. Köhler
    orcid: 0000-0002-5219-1289
    affiliation: 1
  - name: Stefan Haufe
    orcid: 0000-0003-1470-9195
    affiliation: "2, 3, 4, 5, 7"
affiliations:
 - name: Movement Disorders Unit, Charité - Universitätsmedizin Berlin, Germany
   index: 1
 - name: Einstein Center for Neurosciences Berlin, Charité - Universitätsmedizin Berlin, Germany
   index: 2
 - name: Bernstein Center for Computational Neuroscience Berlin, Germany
   index: 3
 - name: Berlin Center for Advanced Neuroimaging, Charité - Universitätsmedizin Berlin, Germany
   index: 4
 - name: Electrical Engineering and Computer Science Department, Technische Universität Berlin, Germany
   index: 5
 - name: Donders Institute for Brain, Cognition and Behaviour, Radboud Universiteit, The Netherlands
   index: 6
 - name: Physikalisch-Technische Bundesanstalt Braunschweig und Berlin, Germany
   index: 7
date: 9 March 2025
bibliography: paper.bib
---

# Summary

Various forms of information can be extracted from neural time series data. Of this, phase-amplitude coupling, time delays, and non-sinusoidal waveshape characteristics are of great interest, providing insights into neuronal function and dysfunction. However, methods commonly used for these analyses possess notable limitations. Recent work has revealed the bispectrum to be a powerful tool for the analysis of electrophysiology data, overcoming many such limitations. Here we present `PyBispectra`, a package for bispectral analyses of electrophysiology data including phase-amplitude coupling, time delays, and non-sinusoidal waveshape.

# Statement of need

Analysis of phase-amplitude coupling, time delays, and non-sinusoidal waveshape provide important insights into interneuronal communication [@Canolty2010;@Silchenko2010;@Sherman2016]. Studies of these features in neural data have been used to investigate core functions such as movement and memory, including their perturbation in disease [@deHemptinne2013;@Cole2017;@Bazzigaluppi2018;@Binns2024]. However, traditional analysis methods have critical limitations that hinder their utility. In contrast, the bispectrum - the Fourier transform of the third order moment [@Nikias1987] - can be used for the analysis of phase-amplitude coupling [@Zandvoort2021], non-sinusoidal waveshape [@Bartz2019], and time delays [@Nikias1988], overcoming many traditional limitations.

Despite these benefits, the bispectrum has seen little use in neuroscience research, in part due to the lack of an accessible toolbox tailored to electrophysiology data. Code written in MATLAB exists for some analyses (see e.g., [github.com/sccn/roiconnect](https://github.com/sccn/roiconnect), [github.com/ZuseDre1/AnalyzingWaveshapeWithBicoherence](https://github.com/ZuseDre1/AnalyzingWaveshapeWithBicoherence)), however it is spread across multiple repositories and often not as toolboxes. Furthermore, this requires a paid MATLAB license, limiting its accessibility. Code for computing the bispectrum exists in the free-to-use Python language [@Bachetti2024], however these implementations are not tailored for electrophysiology data, and while other Python packages perform some of these analyses on electrophysiology data [@Cole2019;@Denker2024], they do not make use of the bispectrum. The `PyBispectra` package addresses this by providing a comprehensive toolbox for bispectral analysis of electrophysiology data (\autoref{fig:overview}), including tutorials to facilitate an understanding of these analyses.

![\label{fig:overview}Overview of the `PyBispectra` toolbox. Optional preprocessing methods are supported for the multivariate analysis of waveshape. Tools are provided for computing spectral representations of time series data, as well as for computing cross-frequency coupling, time delays, and non-sinusoidal waveshape, with schematic visualisations of results shown. Also shown is an example code snippet for analysing phase-amplitude coupling.](Overview.svg)

# Features

## Phase-amplitude coupling

Phase-amplitude coupling is the interaction between the phase of a lower frequency oscillation and amplitude of a higher frequency oscillation. It has been posited as a mechanism for the integration of neural information across spatiotemporal scales [@Canolty2010], with perturbations in disease [@deHemptinne2013;@Bazzigaluppi2018]. Common methods for quantifying phase-amplitude coupling involve bandpass filtering signals in the frequency bands of interest and using the Hilbert transform to extract phase and amplitude information [@Canolty2006;@Tort2010], with several limitations. First, the bandpass filters require precise properties that are not readily apparent, with poorly designed filters smearing information across a broad spectral range [@Zandvoort2021]. Second, the Hilbert transform is a relatively demanding procedure, contributing to a high computational cost. Finally, when analysing interactions between signals, spurious coupling estimates can arise due to interactions within each signal [@PellegriniPreprint]. In contrast, bandpass filtering is not required with the bispectrum, preserving the spectral resolution and reducing the risk of misinterpreting results [@Zandvoort2021]. Furthermore, bispectral analysis relies on the computationally cheap Fourier transform. Finally, spurious across-signal coupling estimates can be corrected for using bispectral antisymmetrisation [@Chella2014;@PellegriniPreprint]. `PyBispectra` provides tools for performing bispectral phase-amplitude coupling, with options for antisymmetrisation and a univariate normalisation procedure that bounds coupling scores in a standardised range for improved interpretability [@Shahbazi2014].

## Time delays

Time delay analysis identifies latencies of information transfer between signals, providing  insight into the physical connections between brain regions [@Silchenko2010;@Binns2024]. A traditional analysis method is cross-correlation, quantifying the similarity of signals at a set of time lags. However, this approach has a limited robustness to noise [@Nikias1988] and a vulnerability to spurious zero time lag interactions arising due to volume conduction and source mixing in the sensor space. On the other hand, the bispectrum is resilient to Gaussian noise [@Nikias1988], and antisymmetrisation can be used to correct for spurious zero time lag interactions [@Chella2014;@JurharPreprint]. `PyBispectra` provides tools for bispectral time delay analysis, with options for antisymmetrisation.

## Non-sinusoidal waveshape

Non-sinusoidal signals indicate properties of interneuronal communication [@Sherman2016], with perturbations seen in disease [@Cole2017]. Various features can be identified, including sawtooth signals and a dominance of peaks or troughs. Analysis can be performed on time series data using peak finding-based procedures - see e.g., @Cole2017 - however this is computationally demanding for high sampling rate data. A further complication comes from the desire to isolate frequency-specific neural activity, with bandpass filtering suppressing non-sinusoidal information [@Bartz2019]. Attempts to address this remain limited by a risk of contamination from frequencies outside the band of interest - see e.g., @Cole2017. In contrast, the bispectrum captures frequency-resolved non-sinusoidal information directly [@Bartz2019] in a computationally efficient manner. `PyBispectra` provides tools for analysing non-sinusoidal waveshape using the bispectrum, including the option of univariate normalisation to bound values in a standardised range for improved interpretability [@Shahbazi2014].

## Supplementary features

Two common issues faced when analysing electrophysiology data are a limited signal-to-noise ratio and interpreting high-dimensional data [@Cohen2022]. Spatio-spectral decomposition is a multivariate technique that addresses these problems, capturing key aspects of frequency-specific information in a high signal-to-noise ratio, low-dimensional space [@Nikulin2011]. This decomposition is supported by `PyBispectra` for the analysis of non-sinusoidal waveshape, with extensions like harmonic power maximisation targetting non-sinusoidal information [@Bartz2019].

Other features of `PyBispectra` include plotting tools for the visualisation of results, low-level compilation with Numba [@Lam2015], and support for parallel processing. Data formats follow conventions from popular signal processing packages like `MNE-Python` [@Gramfort2013], and helper functions are provided as wrappers around `MNE-Python` and `SciPy` [@Virtanen2020] tools to facilitate processing prior to bispectral analyses. Furthermore, tools for amplitude-amplitude and phase-phase coupling are also provided, following literature recommendations for identifying genuine phase-amplitude coupling [@Giehl2021]. Finally, analyses are accompanied by detailed tutorials, facilitating an understanding of how the bispectrum can be used to analyse electrophysiology data.

# Conclusion

Altogether, the bispectrum is a robust and computationally efficient tool for the analysis of phase-amplitude coupling, time delays, and non-sinusoidal waveshape. Bispectral approaches overcome key limitations of traditional methods which have hindered neuroscience research. To aid the uptake of bispectral methods, `PyBispectra` provides access to these tools in a comprehensive, easy-to-use package, tailored for use with electrophysiology data.

# Acknowledgements

We acknowledge contributions from Mr. Toni M. Brotons and Dr. Timon Merk, who provided valuable feedback and suggestions for the design of the `PyBispectra` package and its documentation.

# References