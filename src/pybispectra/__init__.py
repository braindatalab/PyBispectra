__version__ = "0.0.1"

from .cfc import AAC, PAC, PPC
from .tde import TDE
from .utils import (
    ResultsCFC,
    ResultsTDE,
    SpatioSpectralFilter,
    compute_fft,
    fast_find_first,
)
from .waveshape import WaveShape
