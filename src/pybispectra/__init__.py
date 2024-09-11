"""Initialisation of the PyBispectra package."""

__version__ = "1.2.0dev"

from .cfc import AAC, PAC, PPC
from .data import get_example_data_paths
from .general import Bispectrum, Threenorm
from .tde import TDE
from .utils import (
    ResultsCFC,
    ResultsGeneral,
    ResultsTDE,
    ResultsWaveShape,
    SpatioSpectralFilter,
    compute_fft,
    compute_rank,
    compute_tfr,
    set_precision,
)
from .waveshape import WaveShape
