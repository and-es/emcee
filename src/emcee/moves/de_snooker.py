from __future__ import annotations

from typing import Any

import numpy as np

from .red_blue import RedBlueMove

__all__ = ["DESnookerMove"]


class DESnookerMove(RedBlueMove):
    """A snooker proposal using differential evolution.

    Based on `Ter Braak & Vrugt (2008)
    <http://link.springer.com/article/10.1007/s11222-008-9104-9>`_.

    Credit goes to GitHub user `mdanthony17 <https://github.com/mdanthony17>`_
    for proposing this as an addition to the original emcee package.

    Args:
        gammas (Optional[float]): The mean stretch factor for the proposal
            vector. By default, it is :math:`1.7` as recommended by the
            reference.

    """

    gammas: float

    def __init__(self, gammas: float = 1.7, **kwargs: Any) -> None:
        self.gammas = gammas
        kwargs["nsplits"] = 4
        super().__init__(**kwargs)

    def get_proposal(
        self,
        sample: np.ndarray,
        complement: list[np.ndarray],
        random: np.random.Generator,
    ) -> tuple[np.ndarray, np.ndarray]:
        s, c = sample, complement
        Ns, ndim = s.shape
        # Pick one walker from each complementary sub-ensemble, then
        # shuffle the picks within each row with an independent random
        # permutation per row (via argsort of uniform draws).
        w = np.stack(
            [cj[random.integers(len(cj), size=Ns)] for cj in c], axis=1
        )
        perm = np.argsort(random.random((Ns, len(c))), axis=1)
        w = np.take_along_axis(w, perm[:, :, None], axis=1)
        z, z1, z2 = w[:, 0], w[:, 1], w[:, 2]
        delta = s - z
        norm = np.linalg.norm(delta, axis=1)
        # A walker that coincides with its picked ``z`` has no snooker
        # direction to move along; keep it in place and force rejection
        # instead of dividing by zero.
        degenerate = norm == 0
        norm = np.where(degenerate, 1.0, norm)
        u = delta / norm[:, None]
        proj = np.einsum("nd,nd->n", u, z1 - z2)
        q = s + u * (self.gammas * proj)[:, None]
        qz_norm = np.where(degenerate, 1.0, np.linalg.norm(q - z, axis=1))
        factors = (ndim - 1.0) * (np.log(qz_norm) - np.log(norm))
        factors[degenerate] = -np.inf
        return q, factors
