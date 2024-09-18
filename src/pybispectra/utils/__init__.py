"""Helper tools for processing and storing results."""

__version__ = "1.2.0"

from .ged import SpatioSpectralFilter
from .results import ResultsCFC, ResultsGeneral, ResultsTDE, ResultsWaveShape
from .utils import compute_fft, compute_rank, compute_tfr, set_precision
