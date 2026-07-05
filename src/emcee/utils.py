# -*- coding: utf-8 -*-

import warnings
from functools import wraps

import numpy as np

__all__ = [
    "sample_ball",
    "deprecated",
    "deprecation_warning",
    "ensure_rng",
]

# Bit generators that can be reconstructed by name when restoring a
# serialized generator state
_BIT_GENERATORS = ("MT19937", "PCG64", "PCG64DXSM", "Philox", "SFC64")


def deprecation_warning(msg):
    warnings.warn(msg, category=DeprecationWarning, stacklevel=2)


def ensure_rng(rng=None):
    """Coerce a seed or generator into a ``numpy.random.Generator``

    Args:
        rng: Either ``None`` (a fresh unseeded generator), an integer or
            ``numpy.random.SeedSequence`` seed, a ``numpy.random.Generator``
            or ``numpy.random.BitGenerator`` (used as-is), or a legacy
            ``numpy.random.RandomState`` (deprecated; its Mersenne Twister
            state is transferred to the returned generator).

    Returns:
        numpy.random.Generator: The coerced random number generator.

    """
    if isinstance(rng, np.random.RandomState):
        deprecation_warning(
            "'numpy.random.RandomState' is deprecated as a source of "
            "randomness; use 'numpy.random.Generator' instead"
        )
        return random_state_to_generator(rng.get_state())
    return np.random.default_rng(rng)


def random_state_to_generator(state):
    """Convert a legacy ``RandomState.get_state()`` tuple to a ``Generator``

    The Gaussian cache elements of the legacy state tuple are discarded
    because ``numpy.random.Generator`` does not use them.

    Args:
        state: The legacy state, a tuple or list of the form
            ``('MT19937', keys, pos, has_gauss, cached_gaussian)``.

    Returns:
        numpy.random.Generator: A generator backed by an ``MT19937`` bit
        generator carrying the legacy state.

    """
    bit_generator = np.random.MT19937()
    bit_generator.state = {
        "bit_generator": "MT19937",
        "state": {
            "key": np.asarray(state[1], dtype=np.uint32),
            "pos": int(state[2]),
        },
    }
    return np.random.Generator(bit_generator)


def generator_from_state(state):
    """Reconstruct a ``numpy.random.Generator`` from a state dict

    Args:
        state (dict): A bit generator state dict, as returned by
            ``Generator.bit_generator.state``.

    Returns:
        numpy.random.Generator: A generator carrying the given state.

    """
    name = state["bit_generator"]
    if name not in _BIT_GENERATORS:
        raise ValueError(f"unknown bit generator: {name!r}")
    bit_generator = getattr(np.random, name)()
    bit_generator.state = state
    return np.random.Generator(bit_generator)


def deprecated(alternate):
    def wrapper(func, alternate=alternate):
        msg = "'{0}' is deprecated.".format(func.__name__)
        if alternate is not None:
            msg += " Use '{0}' instead.".format(alternate)

        @wraps(func)
        def f(*args, **kwargs):
            deprecation_warning(msg)
            return func(*args, **kwargs)

        return f

    return wrapper


@deprecated(None)
def sample_ball(p0, std, size=1):
    """
    Produce a ball of walkers around an initial parameter value.

    :param p0: The initial parameter value.
    :param std: The axis-aligned standard deviation.
    :param size: The number of samples to produce.

    """
    assert len(p0) == len(std)
    return np.vstack(
        [p0 + std * np.random.normal(size=len(p0)) for i in range(size)]
    )


@deprecated(None)
def sample_ellipsoid(p0, covmat, size=1):
    """
    Produce an ellipsoid of walkers around an initial parameter value,
    according to a covariance matrix.

    :param p0: The initial parameter value.
    :param covmat:
        The covariance matrix.  Must be symmetric-positive definite or
        it will raise the exception numpy.linalg.LinAlgError
    :param size: The number of samples to produce.

    """
    return np.random.multivariate_normal(
        np.atleast_1d(p0), np.atleast_2d(covmat), size=size
    )
