from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING, Any, cast

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping, Sequence

    from numpy.typing import ArrayLike

__all__ = ["State"]


def _copy_value(x: Any) -> Any:
    # ndarray.copy is much faster than deepcopy for arrays; anything
    # else (e.g. the random_state dict) still needs a real deep copy
    if isinstance(x, np.ndarray):
        return x.copy()
    return deepcopy(x)


class State:
    """The state of the ensemble during an MCMC run

    For backwards compatibility, this will unpack into ``coords, log_prob,
    (blobs), random_state`` when iterated over (where ``blobs`` will only be
    included if it exists and is not ``None``).

    Args:
        coords (ndarray[nwalkers, ndim]): The current positions of the walkers
            in the parameter space.
        log_prob (ndarray[nwalkers, ndim], Optional): Log posterior
            probabilities for the  walkers at positions given by ``coords``.
        blobs (Optional): The metadata “blobs” associated with the current
            position. The value is only returned if ``log_prob_fn`` returns
            blobs too.
        random_state (Optional): The current state of the random number
            generator: the ``bit_generator.state`` dict of a
            ``numpy.random.Generator`` (or a legacy
            ``RandomState.get_state()`` tuple written by an older version of
            emcee).
    """

    coords: np.ndarray
    log_prob: np.ndarray | None
    blobs: np.ndarray | None
    random_state: Mapping[str, Any] | Sequence[Any] | None

    __slots__ = "coords", "log_prob", "blobs", "random_state"

    def __init__(
        self,
        coords: State | ArrayLike,
        log_prob: np.ndarray | None = None,
        blobs: np.ndarray | None = None,
        random_state: Mapping[str, Any] | Sequence[Any] | None = None,
        copy: bool = False,
    ) -> None:
        dc = _copy_value if copy else lambda x: x

        if hasattr(coords, "coords"):
            # Also accept duck-typed state objects, e.g. a ``State``
            # whose class identity was lost to a module reload
            other = cast("State", coords)
            self.coords = dc(other.coords)
            self.log_prob = dc(other.log_prob)
            self.blobs = dc(other.blobs)
            self.random_state = dc(other.random_state)
            return

        self.coords = dc(np.atleast_2d(coords))
        self.log_prob = dc(log_prob)
        self.blobs = dc(blobs)
        self.random_state = dc(random_state)

    def __len__(self) -> int:
        if self.blobs is None:
            return 3
        return 4

    def __repr__(self) -> str:
        return (
            f"State({self.coords}, log_prob={self.log_prob}, "
            f"blobs={self.blobs}, random_state={self.random_state})"
        )

    def __iter__(self) -> Iterator[Any]:
        if self.blobs is None:
            return iter((self.coords, self.log_prob, self.random_state))
        return iter(
            (self.coords, self.log_prob, self.random_state, self.blobs)
        )

    def __getitem__(self, index: int) -> Any:
        if index < 0:
            return self[len(self) + index]
        if index == 0:
            return self.coords
        elif index == 1:
            return self.log_prob
        elif index == 2:
            return self.random_state
        elif index == 3 and self.blobs is not None:
            return self.blobs
        raise IndexError(f"Invalid index '{index}'")
