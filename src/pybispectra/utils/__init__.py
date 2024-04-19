"""Helper tools for processing and storing results."""

__version__ = "1.2.0dev"

from .ged import SpatioSpectralFilter
from .results import ResultsCFC, ResultsTDE, ResultsWaveShape, ResultsGeneral
from .utils import compute_fft, compute_tfr, compute_rank, set_precision
