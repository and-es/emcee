import numpy as np
import pytest

from emcee.utils import (
    deprecated,
    deprecation_warning,
    sample_ball,
    sample_ellipsoid,
)

__all__ = [
    "test_deprecation_warning",
    "test_deprecated_decorator",
    "test_sample_ball",
    "test_sample_ellipsoid",
]


def test_deprecation_warning():
    with pytest.warns(DeprecationWarning, match="something is deprecated"):
        deprecation_warning("something is deprecated")


def test_deprecated_decorator():
    @deprecated("new_func")
    def old_func(x):
        return 2 * x

    with pytest.warns(
        DeprecationWarning, match=r"'old_func' is deprecated. Use 'new_func'"
    ):
        assert old_func(21) == 42


def test_deprecated_decorator_no_alternate():
    @deprecated(None)
    def old_func():
        return "value"

    with pytest.warns(DeprecationWarning, match=r"'old_func' is deprecated."):
        assert old_func() == "value"


def test_sample_ball(seed=1234):
    # The deprecated helper itself draws from the global numpy RNG
    np.random.seed(seed)
    p0 = np.array([1.0, 10.0, -4.0])
    std = np.array([0.1, 0.5, 0.01])
    with pytest.warns(DeprecationWarning):
        ball = sample_ball(p0, std, size=1000)
    assert ball.shape == (1000, len(p0))
    assert np.allclose(np.mean(ball, axis=0), p0, atol=0.1)
    assert np.allclose(np.std(ball, axis=0), std, rtol=0.2)


def test_sample_ellipsoid(seed=1234):
    # The deprecated helper itself draws from the global numpy RNG
    np.random.seed(seed)
    p0 = np.array([1.0, -2.0])
    covmat = np.array([[1.0, 0.5], [0.5, 2.0]])
    with pytest.warns(DeprecationWarning):
        samples = sample_ellipsoid(p0, covmat, size=5000)
    assert samples.shape == (5000, len(p0))
    assert np.allclose(np.mean(samples, axis=0), p0, atol=0.1)
    assert np.allclose(np.cov(samples, rowvar=False), covmat, atol=0.2)
