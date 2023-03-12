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
:ref:`[2] <deHemptinne2013>` and Alzheimer's :ref:`[3] <Bazzigaluppi2017>`.

On the other hand, time delay estimation methods allow us to analyse the
interactions across signals and frequencies by estimating the time delay,
:math:`\tau`, between signals. Estimating time delays is a crucial matter for
radar and sonar technologies :ref:`[4] <Chen2004>`, however it is also of
interest in other signal analysis contexts, such as again in neuroscience
where time delays can be used to infer features of the physical relationships
between interacting brain regions :ref:`[5] <Silchenko2010>`.


What are bispectra and why use them?
------------------------------------




What is available in PyBispectra?
---------------------------------
PyBispectra offers tools for computing phase-amplitude coupling and time delay
estimation with bispectra, as well as for computing phase-phase coupling.
Additional tools are included for plotting results and for computing the
Fourier coefficients required by these methods.

You can find the installation instructions :doc:`here <installation>`, as well
as examples of how tools in the package can be used :doc:`here <examples>`.


References
----------
.. _Canolty2010:

[1] `Canolty & Knight (2010). The functional role of cross-frequency coupling. Trends in Cognitive Sciences. DOI: 10.1016/j.tics.2010.09.001 <https://doi.org/10.1016%2Fj.tics.2010.09.001>`_

.. _deHemptinne2013:

[2] `de Hemptinne et al. (2013). Exaggerated phase-amplitude coupling in the primary motor cortex in Parkinson disease. Proceedings of the National Academy of Sciences. DOI: 10.1073/pnas.1214546110 <https://doi.org/10.1073/pnas.1214546110>`_

.. _Bazzigaluppi2017:

[3] `Bazzigaluppi et al. (2017). Early-stage attenuation of phase-amplitude coupling in the hippocampus and medial prefrontal cortex in a transgenic rat model of Alzheimer's disease. Journal of Neurochemistry. DOI: 10.1111/jnc.14136 <https://doi.org/10.1111/jnc.14136>`_

.. _Chen2004:

[4] `Chen et al. (2004). Time Delay Estimation. In: Huang & Benesty, Audio Signal Processing for Next-Generation Multimedia Communication Systems. Springer. DOI: 10.1007/1-4020-7769-6_8 <https://doi.org/10.1007/1-4020-7769-6_8>`_

.. _Silchenko2010:

[5] `Silchenko et al. (2010). Data-driven approach to the estimation of connectivity and time delays in the coupling of interacting neuronal subsystems. Journal of Neuroscience Methods. DOI: 10.1016/j.jneumeth.2010.06.004 <https://doi.org/10.1016/j.jneumeth.2010.06.004>`_
