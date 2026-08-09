import numpy as np

import emcee

try:
    from scipy import stats
except ImportError:
    stats = None  # ty: ignore[invalid-assignment]


__all__ = ["_test_normal", "_test_uniform"]


def normal_log_prob_blobs(params):
    return -0.5 * np.sum(params**2), params


def normal_log_prob(params):
    return -0.5 * np.sum(params**2)


def uniform_log_prob(params):
    if np.any(params > 1) or np.any(params < 0):
        return -np.inf
    return 0.0


def _test_normal(
    proposal,
    ndim=1,
    nwalkers=32,
    nsteps=2000,
    seed=1234,
    check_acceptance=True,
    pool=None,
    blobs=False,
):
    # Set up the random number generator.
    rng = np.random.default_rng(seed)

    # Initialize the ensemble and proposal.
    coords = rng.standard_normal((nwalkers, ndim))

    if blobs:
        lp = normal_log_prob_blobs
    else:
        lp = normal_log_prob

    sampler = emcee.EnsembleSampler(
        nwalkers, ndim, lp, moves=proposal, pool=pool, rng=rng
    )
    sampler.run_mcmc(coords, nsteps)

    # Check the acceptance fraction.
    if check_acceptance:
        acc = sampler.acceptance_fraction
        assert np.all((acc < 0.9) * (acc > 0.1)), (
            f"Invalid acceptance fraction\n{acc}"
        )

    # Check the resulting chain using a K-S test and compare to the mean and
    # standard deviation.
    samps = sampler.get_chain(flat=True)
    mu, sig = np.mean(samps, axis=0), np.std(samps, axis=0)
    assert np.all(np.abs(mu) < 0.08), "Incorrect mean"
    assert np.all(np.abs(sig - 1) < 0.05), "Incorrect standard deviation"

    if ndim == 1 and stats is not None:
        ks, _ = stats.kstest(samps[:, 0], "norm")
        assert ks < 0.05, "The K-S test failed"


def _test_uniform(proposal, nwalkers=32, nsteps=2000, seed=1234):
    # Set up the random number generator.
    rng = np.random.default_rng(seed)

    # Initialize the ensemble and proposal.
    coords = rng.random((nwalkers, 1))

    sampler = emcee.EnsembleSampler(
        nwalkers, 1, uniform_log_prob, moves=proposal, rng=rng
    )
    sampler.run_mcmc(coords, nsteps)

    # Check the acceptance fraction.
    acc = sampler.acceptance_fraction
    assert np.all((acc < 0.95) * (acc > 0.1)), (
        f"Invalid acceptance fraction\n{acc}"
    )

    # Compare the sample mean and standard deviation to the expected
    # moments of U(0, 1).
    samps = sampler.get_chain(flat=True)
    mu, sig = np.mean(samps), np.std(samps)
    assert np.abs(mu - 0.5) < 0.05, "Incorrect mean"
    assert np.abs(sig - 1.0 / np.sqrt(12)) < 0.05, (
        "Incorrect standard deviation"
    )

    if stats is not None:
        # Check the (thinned) chain against the target using a K-S test.
        ks, _ = stats.kstest(samps[::100, 0], "uniform")
        assert ks < 0.1, "The K-S test failed"
