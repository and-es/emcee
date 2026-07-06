import warnings
from functools import wraps

import numpy as np

__all__ = [
    "deprecated",
    "deprecation_warning",
    "ensure_rng",
]


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
    return generator_from_state(
        {
            "bit_generator": "MT19937",
            "state": {
                "key": np.asarray(state[1], dtype=np.uint32),
                "pos": int(state[2]),
            },
        }
    )


def generator_from_state(state):
    """Reconstruct a ``numpy.random.Generator`` from a state dict

    Args:
        state (dict): A bit generator state dict, as returned by
            ``Generator.bit_generator.state``. The named bit generator
            must be one provided by ``numpy.random``; third-party bit
            generators cannot be reconstructed by name.

    Returns:
        numpy.random.Generator: A generator carrying the given state.

    """
    name = state["bit_generator"]
    bit_generator_cls = getattr(np.random, name, None)
    if not (
        isinstance(bit_generator_cls, type)
        and issubclass(bit_generator_cls, np.random.BitGenerator)
    ):
        raise ValueError(f"unknown bit generator: {name!r}")
    bit_generator = bit_generator_cls()
    bit_generator.state = state
    return np.random.Generator(bit_generator)


def stored_state_to_generator(state):
    """Convert a stored random state to a ``numpy.random.Generator``

    Args:
        state: Either a bit generator state dict (as returned by
            ``Generator.bit_generator.state``) or a legacy
            ``RandomState.get_state()`` tuple/list written by an older
            version of emcee.

    Returns:
        numpy.random.Generator: A generator carrying the given state.

    """
    if isinstance(state, dict):
        return generator_from_state(state)
    return random_state_to_generator(state)


def deprecated(alternate):
    def wrapper(func, alternate=alternate):
        msg = f"'{func.__name__}' is deprecated."
        if alternate is not None:
            msg += f" Use '{alternate}' instead."

        @wraps(func)
        def f(*args, **kwargs):
            deprecation_warning(msg)
            return func(*args, **kwargs)

        return f

    return wrapper
