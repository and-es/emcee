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
            it will use all the walkers in the complement.

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
        q = np.empty_like(s)
        s0 = Nc if self.s is None else self.s
        for i in range(Ns):
            inds = random.choice(Nc, s0, replace=False)
            cov = np.atleast_2d(np.cov(c[inds], rowvar=False))
            q[i] = random.multivariate_normal(s[i], cov)
        return q, np.zeros(Ns, dtype=np.float64)
