from __future__ import annotations

from typing import Any

import numpy as np

from .red_blue import RedBlueMove

__all__ = ["WalkMove"]


class WalkMove(RedBlueMove):
    """
    A `Goodman & Weare (2010)
    <https://msp.org/camcos/2010/5-1/p04.xhtml>`_ "walk move" with
    parallelization as described in `Foreman-Mackey et al. (2013)
    <https://arxiv.org/abs/1202.3665>`_.

    Args:
        s (Optional[int]): The number of helper walkers to use. By default
            it will use all the walkers in the complement. Must be at
            least 2 and no larger than the number of complementary
            walkers.

    """

    s: int | None

    def __init__(self, s: int | None = None, **kwargs: Any) -> None:
        self.s = s
        super().__init__(**kwargs)

    def get_proposal(
        self,
        sample: np.ndarray,
        complement: list[np.ndarray],
        random: np.random.Generator,
    ) -> tuple[np.ndarray, np.ndarray]:
        s = sample
        c = np.concatenate(complement, axis=0)
        Ns, Nc = len(s), len(c)
        s0 = Nc if self.s is None else self.s
        if not 2 <= s0 <= Nc:
            raise ValueError(
                f"'s' must be between 2 and the number of complementary "
                f"walkers ({Nc}); got {s0}"
            )

        # For iid standard normal z_j, the perturbation
        # sum_j z_j * (c_j - mean(c)) / sqrt(s0 - 1) is Gaussian with
        # covariance exactly equal to the sample covariance of the helper
        # subset, so this matches sampling from a multivariate normal
        # centered on each walker with that covariance.
        if s0 == Nc:
            # The subset is always the full complement, so no per-walker
            # subset selection is needed.
            centered = c - c.mean(axis=0)
            z = random.standard_normal((Ns, Nc))
            q = s + z @ centered / np.sqrt(Nc - 1)
        else:
            # Sample without replacement for every walker at once by
            # taking the first s0 entries of an independent random
            # permutation per row.
            inds = np.argsort(random.random((Ns, Nc)), axis=1)[:, :s0]
            subs = c[inds]  # (Ns, s0, ndim)
            centered = subs - subs.mean(axis=1, keepdims=True)
            z = random.standard_normal((Ns, s0))
            q = s + np.einsum("ns,nsd->nd", z, centered) / np.sqrt(s0 - 1)
        return q, np.zeros(Ns, dtype=np.float64)
