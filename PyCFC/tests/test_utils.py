"""Tests for toolbox utilities."""

import pytest
import numpy as np

from pycfc import fast_find_first


def test_fast_find_first_present_value():
    """Test that `fast_find_first` finds a present value."""
    index = fast_find_first(vector=np.array([-1, 0, 1, 2, 3, 1]), value=1)
    assert index == 2, "The index of the value being found should be 2."


def test_fast_find_first_missing_value():
    """Test that `fast_find_first` raises an error for a missing value."""
    with pytest.raises(
        ValueError, match="`value` is not present in `vector`."
    ):
        fast_find_first(vector=np.array([-1, 0, 2, 2, 3, 4]), value=1)
