"""Default values for the PyBispectra package."""

import numpy as np


class _Precision:
    """Class specifying precision of data/results.

    Double precision (i.e. float64 and complex128) used by default.
    """

    def __init__(self) -> None:  # noqa: D107
        self.type = "double"
        self.real = np.float64
        self.complex = np.complex128

    def set_precision(self, precision: str) -> None:
        """Set precision of data/results.

        Parameters
        ----------
        precision : str
            Precision of data/results. Must be one of "single" or "double".
        """
        if precision not in ["single", "double"]:
            raise ValueError("precision must be either 'single' or 'double'.")

        if precision == "single":
            self.type = "single"
            self.real = np.float32
            self.complex = np.complex64
        else:
            self.type = "double"
            self.real = np.float64
            self.complex = np.complex128


_precision = _Precision()
