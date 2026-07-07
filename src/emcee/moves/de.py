from __future__ import annotations

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
        ns = len(s)
        nc = len(c)

        # Draw an ordered pair of distinct complementary walkers for
        # each walker: ``i`` uniform, then ``j`` uniform over the
        # remaining indices. This samples the same distribution as
        # enumerating all non-diagonal index pairs, without the
        # O(nc^2) pair table.
        i = random.integers(nc, size=ns)
        j = random.integers(nc - 1, size=ns)
        j += j >= i

        # Compute diff vectors
        diffs = c[j] - c[i]  # (ns, ndim)

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
