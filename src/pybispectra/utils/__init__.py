__version__ = "0.0.1"

from .utils import (
    _ResultsBase,
    ResultsCFC,
    ResultsTDE,
    compute_fft,
    fast_find_first,
    _generate_data,
)
from .process import _ProcessBase, _ProcessBispectra, _compute_bispectrum
