"""Helper tools for processing and storing results."""

__version__ = "0.0.1"

from .utils import (
    ResultsCFC,
    ResultsTDE,
    compute_fft,
    fast_find_first,
    _generate_data,
)
from ._process import _ProcessBase, _ProcessBispectra, _compute_bispectrum
from ._docs import linkcode_resolve
