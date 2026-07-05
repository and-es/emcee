import numpy as np
import pytest

import emcee
from emcee import moves

from .test_proposal import _test_normal, normal_log_prob

__all__ = ["test_normal_mh", "test_mh_ndim_mismatch"]


def gaussian_proposal(coords, random):
    # A symmetric Gaussian random-walk proposal with zero log-ratio.
    return (
        coords + 1.0 * random.standard_normal(coords.shape),
        np.zeros(len(coords)),
    )


@pytest.mark.parametrize("blobs", [True, False])
def test_normal_mh(blobs, **kwargs):
    _test_normal(moves.MHMove(gaussian_proposal), blobs=blobs, **kwargs)


def test_mh_ndim_mismatch(seed=1234):
    nwalkers, ndim = 32, 3
    coords = np.random.default_rng(seed).standard_normal((nwalkers, ndim))
    sampler = emcee.EnsembleSampler(
        nwalkers,
        ndim,
        normal_log_prob,
        moves=moves.MHMove(gaussian_proposal, ndim=ndim + 1),
    )
    with pytest.raises(ValueError, match="Dimension mismatch"):
        sampler.run_mcmc(coords, 1)
