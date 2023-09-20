"""Helper tools for processing and storing results."""

__version__ = "1.0.0"

from .ged import SpatioSpectralFilter
from .results import ResultsCFC, ResultsTDE, ResultsWaveShape
from .utils import compute_fft, compute_tfr, compute_rank
