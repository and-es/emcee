import numpy as np
import pytest

from emcee import moves
from emcee.moves.move import Move
from emcee.state import State

__all__ = [
    "test_update_requires_matching_blobs",
    "test_gaussian_invalid_cov_shape",
    "test_gaussian_invalid_factor",
]


def test_update_requires_matching_blobs():
    old = State(np.zeros((4, 2)), log_prob=np.zeros(4))
    new = State(np.ones((4, 2)), log_prob=np.ones(4), blobs=np.ones(4))
    accepted = np.ones(4, dtype=bool)
    with pytest.raises(ValueError, match="current list of blobs"):
        Move().update(old, new, accepted)


def test_gaussian_invalid_cov_shape():
    # A non-square matrix...
    with pytest.raises(ValueError, match="Invalid proposal scale dimensions"):
        moves.GaussianMove(np.zeros((2, 3)))

    # ... and too many dimensions are both invalid.
    with pytest.raises(ValueError, match="Invalid proposal scale dimensions"):
        moves.GaussianMove(np.zeros((2, 2, 2)))


def test_gaussian_invalid_factor():
    with pytest.raises(ValueError, match="'factor' must be >= 1.0"):
        moves.GaussianMove(1.0, factor=0.5)
