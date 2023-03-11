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
neuronal communication in the brain :ref:`[1] <Canolty2010>`, with alterations
in these relationships implicated in diseases such as Parkinson's
:ref:`[2] <deHemptinne2013>` and Alzheimer's :ref:`[3] <Canolty2010>`.

On the other hand, time delay estimation methods allow us to analyse the
interactions across signals and frequencies by estimating the time delay,
:math:`\tau`, between signals. Estimating time delays has been used for...


What are bispectra and why use them?
------------------------------------




What is available in PyBispectra?
---------------------------------
PyBispectra offers tools for computing phase-amplitude coupling and time delay
estimation with bispectra, as well as for computing phase-phase coupling.
Additional tools are included for plotting results and for computing the
Fourier coefficients required by these methods.

Check out the :doc:`usage` guide for installation instructions and examples of
how the package can be used.


References
----------
.. _Canolty2010:

[1] `Canolty & Knight (2010). The functional role of cross-frequency coupling. Trends in Cognitive Sciences. DOI: 10.1016/j.tics.2010.09.001 <https://doi.org/10.1016%2Fj.tics.2010.09.001>`_

.. _deHemptinne2013:

[2] `de Hemptinne et al. (2013). Exaggerated phase-amplitude coupling in the primary motor cortex in Parkinson disease. Proceedings of the National Academy of Sciences. DOI: 10.1073/pnas.1214546110 <https://doi.org/10.1073/pnas.1214546110>`_

.. _Bazzigaluppi2017:

[3] `Bazzigaluppi et al. (2017). Early-stage attenuation of phase-amplitude coupling in the hippocampus and medial prefrontal cortex in a transgenic rat model of Alzheimer's disease. Journal of Neurochemistry. DOI: 10.1111/jnc.14136 <https://doi.org/10.1111/jnc.14136>`_