"""Initialisation of the PyBispectra package."""

__version__ = "1.2.0dev"

from .cfc import AAC, PAC, PPC
from .tde import TDE
from .waveshape import WaveShape
from .general import Bispectrum, Threenorm
from .utils import (
    ResultsCFC,
    ResultsTDE,
    ResultsWaveShape,
    ResultsGeneral,
    SpatioSpectralFilter,
    compute_fft,
    compute_tfr,
    compute_rank,
    set_precision,
)
from .data import get_example_data_paths
