import numpy as np
import pytest

from emcee import EnsembleSampler
from emcee.state import State


def check_rstate(a, b):
    # Bit generator state dicts of a PCG64 generator compare directly
    assert a == b


def test_back_compat(seed=1234):
    rng = np.random.default_rng(seed)
    coords = rng.standard_normal((16, 3))
    log_prob = rng.standard_normal(len(coords))
    blobs = rng.standard_normal(len(coords))
    rstate = rng.bit_generator.state

    state = State(coords, log_prob, blobs, rstate)
    c, lp, r, b = state
    assert np.allclose(coords, c)
    assert np.allclose(log_prob, lp)
    assert np.allclose(blobs, b)
    check_rstate(rstate, r)

    state = State(coords, log_prob, None, rstate)
    c, lp, r = state
    assert np.allclose(coords, c)
    assert np.allclose(log_prob, lp)
    check_rstate(rstate, r)


def test_overwrite(seed=1234):
    def ll(x):
        return -0.5 * np.sum(x**2)

    nwalkers = 64
    p0 = np.random.default_rng(seed).normal(size=(nwalkers, 1))
    init = np.copy(p0)

    sampler = EnsembleSampler(nwalkers, 1, ll)
    sampler.run_mcmc(p0, 10)
    assert np.allclose(init, p0)


def test_indexing(seed=1234):
    rng = np.random.default_rng(seed)
    coords = rng.standard_normal((16, 3))
    log_prob = rng.standard_normal(len(coords))
    blobs = rng.standard_normal(len(coords))
    rstate = rng.bit_generator.state

    state = State(coords, log_prob, blobs, rstate)
    np.testing.assert_allclose(state[0], state.coords)
    np.testing.assert_allclose(state[1], state.log_prob)
    check_rstate(state[2], state.random_state)
    np.testing.assert_allclose(state[3], state.blobs)
    np.testing.assert_allclose(state[-1], state.blobs)
    with pytest.raises(IndexError):
        state[4]

    state = State(coords, log_prob, random_state=rstate)
    np.testing.assert_allclose(state[0], state.coords)
    np.testing.assert_allclose(state[1], state.log_prob)
    check_rstate(state[2], state.random_state)
    check_rstate(state[-1], state.random_state)
    with pytest.raises(IndexError):
        state[3]
