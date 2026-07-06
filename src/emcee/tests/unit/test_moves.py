import numpy as np
import pytest

from emcee import moves
from emcee.moves.de import _get_nondiagonal_pairs
from emcee.moves.move import Move
from emcee.state import State

__all__ = [
    "test_update_requires_matching_blobs",
    "test_gaussian_invalid_cov_shape",
    "test_gaussian_invalid_factor",
    "test_nondiagonal_pairs_cache_is_read_only",
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


def test_nondiagonal_pairs_cache_is_read_only():
    # The result is cached and shared, so mutating it must fail instead of
    # silently polluting the cache for subsequent callers.
    pairs = _get_nondiagonal_pairs(5)
    with pytest.raises(ValueError, match="read-only"):
        pairs[0, 0] = -1

    # A second call must return the same, unmodified pairs.
    expected = [(i, j) for i in range(5) for j in range(5) if i != j]
    assert sorted(map(tuple, _get_nondiagonal_pairs(5))) == sorted(expected)
