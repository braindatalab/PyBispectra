"""Initialisation of the PyBispectra package."""

__version__ = "1.3.1+dev"

from .cfc import AAC, PAC, PPC
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
    get_example_data_paths,
    set_precision,
)
from .waveshape import WaveShape
