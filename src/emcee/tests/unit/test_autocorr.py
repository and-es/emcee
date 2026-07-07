import warnings

import numpy as np
import pytest

from emcee.autocorr import (
    AutocorrError,
    auto_window,
    function_1d,
    integrated_time,
)


def get_chain(seed=1234, ndim=3, N=100000):
    rng = np.random.default_rng(seed)
    a = 0.9
    x = np.empty((N, ndim))
    x[0] = np.zeros(ndim)
    for i in range(1, N):
        x[i] = x[i - 1] * a + rng.random(ndim)
    return x


def test_1d(seed=1234, ndim=1, N=250000):
    x = get_chain(seed=seed, ndim=ndim, N=N)
    tau = integrated_time(x)
    assert np.all(np.abs(tau - 19.0) / 19.0 < 0.2)


def test_nd(seed=1234, ndim=3, N=150000):
    x = get_chain(seed=seed, ndim=ndim, N=N)
    tau = integrated_time(x)
    assert np.all(np.abs(tau - 19.0) / 19.0 < 0.2)


def test_nd_without_walkers(seed=1234, ndim=3, N=10000):
    x = get_chain(seed=seed, ndim=ndim, N=N)
    tau1 = integrated_time(x[:, np.newaxis])
    tau2 = integrated_time(x, has_walkers=False)
    assert np.allclose(tau1, tau2)


def test_too_short(seed=1234, ndim=3, N=100):
    x = get_chain(seed=seed, ndim=ndim, N=N)
    with pytest.raises(AutocorrError):
        integrated_time(x)
    tau = integrated_time(x, quiet=True)  # NOQA


def test_function_1d_invalid_dimensions():
    with pytest.raises(ValueError, match="invalid dimensions"):
        function_1d(np.zeros((10, 2)))


def test_invalid_dimensions():
    with pytest.raises(ValueError, match="invalid dimensions"):
        integrated_time(np.zeros((10, 2, 3, 4)))


def test_auto_window_no_crossing():
    # If the window condition is never satisfied, auto_window must fall
    # back to the largest available window.
    assert auto_window(np.zeros(10), 5) == 9


def test_constant_walker(seed=5, nsteps=2000, nwalkers=8, ndim=3):
    # A walker whose chain is constant (e.g. no proposal was ever
    # accepted) has an undefined autocorrelation. This must not leak a
    # numpy RuntimeWarning; it must surface as NaN flagged by the
    # non-convergence check instead.
    x = np.random.default_rng(seed).standard_normal((nsteps, nwalkers, ndim))
    x[:, 5, 0] = 1.0

    with warnings.catch_warnings():
        warnings.simplefilter("error")

        with pytest.raises(AutocorrError, match="undefined"):
            integrated_time(x)

        tau = integrated_time(x, quiet=True)

    assert np.isnan(tau[0])
    assert np.all(np.isfinite(tau[1:]))

    # A fully constant chain gives NaN for every parameter
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        tau = integrated_time(np.ones((nsteps, nwalkers, ndim)), quiet=True)
    assert np.all(np.isnan(tau))


def test_function_1d_constant_series():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        acf = function_1d(np.ones(10))
    assert np.all(np.isnan(acf))


def test_autocorr_multi_works():
    xs = np.random.default_rng(42).standard_normal((16384, 2))

    acls_multi = integrated_time(xs[:, np.newaxis])
    acls_single = np.array(
        [integrated_time(xs[:, i]) for i in range(xs.shape[1])]
    ).squeeze()

    assert np.allclose(acls_multi, acls_single)
