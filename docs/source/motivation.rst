Motivation
==========

Why analyse cross-frequency coupling and time delay estimates?
--------------------------------------------------------------
Cross-frequency coupling and time delay estimation are useful for signal
analysis across a range of disciplines in neuroscience and physics.

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

On the other hand, time delay estimation methods allow us to analyse the
interactions across signals and frequencies by estimating the time delay,
:math:`\tau`, between signals. Estimating time delays is a crucial matter for
radar and sonar technologies :footcite:`Chen2004`, however it is also of
interest in other signal analysis contexts, such as again in neuroscience
where time delays can be used to infer features of the physical relationships
between interacting brain regions :footcite:`Silchenko2010`.


What is the bispectrum and why use it?
--------------------------------------




What is available in PyBispectra?
---------------------------------
PyBispectra offers tools for computing phase-amplitude coupling, time delay
estimation, and wave shape feature analysis with bispectra. Additional tools
are included for computing phase-phase coupling, amplitude-amplitude coupling,
and Fourier coefficients, as well as tools for performing generalised
eigendecompositions and plotting results.

You can find the installation instructions :doc:`here <installation>`, as well
as examples of how tools in the package can be used :doc:`here <examples>`.


References
----------
.. footbibliography::
