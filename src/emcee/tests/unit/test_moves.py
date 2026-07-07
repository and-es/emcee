import numpy as np
import pytest

from emcee import moves
from emcee.model import Model
from emcee.moves.de import _get_nondiagonal_pairs
from emcee.moves.move import Move
from emcee.state import State

__all__ = [
    "test_update_requires_matching_blobs",
    "test_update_requires_log_prob",
    "test_propose_requires_log_prob",
    "test_gaussian_invalid_cov_shape",
    "test_gaussian_invalid_factor",
    "test_nondiagonal_pairs_cache_is_read_only",
    "test_walk_invalid_s",
]


def test_update_requires_matching_blobs():
    old = State(np.zeros((4, 2)), log_prob=np.zeros(4))
    new = State(np.ones((4, 2)), log_prob=np.ones(4), blobs=np.ones(4))
    accepted = np.ones(4, dtype=bool)
    with pytest.raises(ValueError, match="current list of blobs"):
        Move().update(old, new, accepted)


def test_update_requires_log_prob():
    old = State(np.zeros((4, 2)))
    new = State(np.ones((4, 2)), log_prob=np.ones(4))
    accepted = np.ones(4, dtype=bool)
    with pytest.raises(ValueError, match="computed log probabilities"):
        Move().update(old, new, accepted)


@pytest.mark.parametrize(
    "move",
    [
        moves.StretchMove(),
        moves.MHMove(lambda coords, rng: (coords, np.zeros(len(coords)))),
    ],
)
def test_propose_requires_log_prob(move):
    model = Model(
        None,
        lambda x: (np.zeros(len(x)), None),
        map,
        np.random.default_rng(0),
    )
    state = State(np.zeros((10, 2)))
    with pytest.raises(ValueError, match="computed log probabilities"):
        move.propose(model, state)


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


@pytest.mark.parametrize("s", [0, 1, 17])
def test_walk_invalid_s(s):
    # 's' larger than the complement or too small to define a covariance
    # must fail with an explicit error.
    rng = np.random.default_rng(0)
    sample = rng.standard_normal((8, 2))
    complement = [rng.standard_normal((16, 2))]
    with pytest.raises(ValueError, match="'s' must be between 2 and"):
        moves.WalkMove(s=s).get_proposal(sample, complement, rng)


def test_nondiagonal_pairs_cache_is_read_only():
    # The result is cached and shared, so mutating it must fail instead of
    # silently polluting the cache for subsequent callers.
    pairs = _get_nondiagonal_pairs(5)
    with pytest.raises(ValueError, match="read-only"):
        pairs[0, 0] = -1

    # A second call must return the same, unmodified pairs.
    expected = [(i, j) for i in range(5) for j in range(5) if i != j]
    assert sorted(map(tuple, _get_nondiagonal_pairs(5))) == sorted(expected)
