from __future__ import annotations

from functools import lru_cache
from typing import Any

import numpy as np

from .red_blue import RedBlueMove

__all__ = ["DEMove"]


class DEMove(RedBlueMove):
    r"""A proposal using differential evolution.

    This `Differential evolution proposal
    <http://www.stat.columbia.edu/~gelman/stuff_for_blog/cajo.pdf>`_ is
    implemented following `Nelson et al. (2013)
    <https://doi.org/10.1088/0067-0049/210/1/11>`_.

    Args:
        sigma (float): The standard deviation of the Gaussian used to stretch
            the proposal vector.
        gamma0 (Optional[float]): The mean stretch factor for the proposal
            vector. By default, it is :math:`2.38 / \sqrt{2\,\mathrm{ndim}}`
            as recommended by the two references.

    """

    sigma: float
    gamma0: float | None
    g0: float

    def __init__(
        self,
        sigma: float = 1.0e-5,
        gamma0: float | None = None,
        **kwargs: Any,
    ) -> None:
        self.sigma = sigma
        self.gamma0 = gamma0
        super().__init__(**kwargs)

    def setup(self, coords: np.ndarray) -> None:
        if self.gamma0 is None:
            # Pure MAGIC:
            ndim = coords.shape[1]
            self.g0 = 2.38 / np.sqrt(2 * ndim)
        else:
            self.g0 = self.gamma0

    def get_proposal(
        self,
        sample: np.ndarray,
        complement: list[np.ndarray],
        random: np.random.Generator,
    ) -> tuple[np.ndarray, np.ndarray]:
        s = sample
        c = np.concatenate(complement, axis=0)
        ns, ndim = s.shape
        nc = c.shape[0]

        # Get the pair indices
        pairs = _get_nondiagonal_pairs(nc)

        # Sample from the pairs
        indices = random.choice(pairs.shape[0], size=ns, replace=True)
        pairs = pairs[indices]

        # Compute diff vectors
        diffs = np.diff(c[pairs], axis=1).squeeze(axis=1)  # (ns, ndim)

        # Sample a gamma value for each walker following Nelson et al. (2013)
        gamma = self.g0 * (
            1 + self.sigma * random.standard_normal((ns, 1))
        )  # (ns, 1)

        # In this way, sigma is the standard deviation of the distribution
        # of gamma, instead of the standard deviation of the distribution of
        # the proposal as proposed by Ter Braak (2006). Otherwise, sigma
        # should be tuned for each dimension, which confronts the idea of
        # affine-invariance.

        q = s + gamma * diffs

        return q, np.zeros(ns, dtype=np.float64)


# With an odd number of walkers the two complement sizes differ by one
# and alternate within every propose call, so the cache must hold both
# to avoid recomputing the O(n^2) pair table twice per step.
@lru_cache(maxsize=2)
def _get_nondiagonal_pairs(n: int) -> np.ndarray:
    """Get the indices of a square matrix of size n, excluding the
    diagonal."""
    rows, cols = np.tril_indices(n, -1)  # -1 to exclude diagonal

    # Combine rows-cols and cols-rows pairs
    pairs = np.column_stack(
        [np.concatenate([rows, cols]), np.concatenate([cols, rows])]
    )

    # The array is cached and shared between callers, so make it read-only
    # to prevent accidental in-place modification from polluting the cache.
    pairs.setflags(write=False)

    return pairs
