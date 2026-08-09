from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from .mh import MHMove

if TYPE_CHECKING:
    from numpy.typing import ArrayLike

__all__ = ["GaussianMove"]


class GaussianMove(MHMove):
    """A Metropolis step with a Gaussian proposal function.

    Args:
        cov: The covariance of the proposal function. This can be a scalar,
            vector, or matrix and the proposal will be assumed isotropic,
            axis-aligned, or general respectively.
        mode (Optional): Select the method used for updating parameters. This
            can be one of ``"vector"``, ``"random"``, or ``"sequential"``. The
            ``"vector"`` mode updates all dimensions simultaneously,
            ``"random"`` randomly selects a dimension and only updates that
            one, and ``"sequential"`` loops over dimensions and updates each
            one in turn.
        factor (Optional[float]): If provided the proposal will be made with a
            standard deviation uniformly selected from the range
            ``exp(U(-log(factor), log(factor))) * cov``. This is invalid for
            the ``"vector"`` mode.

    Raises:
        ValueError: If the proposal dimensions are invalid or if any of any of
            the other arguments are inconsistent.

    """

    def __init__(
        self,
        cov: ArrayLike,
        mode: str = "vector",
        factor: float | None = None,
    ) -> None:
        # Parse the proposal type.
        ndim: int | None
        try:
            # EAFP probe: only a scalar ``cov`` is float-convertible
            float(cov)  # ty: ignore[invalid-argument-type]

        except TypeError:
            cov = np.atleast_1d(cov)
            if len(cov.shape) == 1:
                # A diagonal proposal was given.
                ndim = len(cov)
                proposal = _diagonal_proposal(np.sqrt(cov), factor, mode)

            elif len(cov.shape) == 2 and cov.shape[0] == cov.shape[1]:
                # The full, square covariance matrix was given.
                ndim = cov.shape[0]
                proposal = _proposal(cov, factor, mode)

            else:
                raise ValueError("Invalid proposal scale dimensions") from None

        else:
            # This was a scalar proposal.
            ndim = None
            proposal = _isotropic_proposal(np.sqrt(cov), factor, mode)

        super().__init__(proposal, ndim=ndim)


class _isotropic_proposal:
    allowed_modes = ["vector", "random", "sequential"]

    index: int
    scale: Any
    mode: str
    _log_factor: float | None

    def __init__(self, scale: Any, factor: float | None, mode: str) -> None:
        self.index = 0
        self.scale = scale
        if factor is None:
            self._log_factor = None
        else:
            if factor < 1.0:
                raise ValueError("'factor' must be >= 1.0")
            self._log_factor = np.log(factor)

        if mode not in self.allowed_modes:
            raise ValueError(
                f"'{mode}' is not a recognized mode. "
                f"Please select from: {self.allowed_modes}"
            )
        self.mode = mode

    def get_factor(self, rng: np.random.Generator) -> float:
        if self._log_factor is None:
            return 1.0
        return np.exp(rng.uniform(-self._log_factor, self._log_factor))

    def get_updated_vector(
        self, rng: np.random.Generator, x0: np.ndarray
    ) -> np.ndarray:
        return x0 + self.get_factor(rng) * self.scale * rng.standard_normal(
            x0.shape
        )

    def __call__(
        self, x0: np.ndarray, rng: np.random.Generator
    ) -> tuple[np.ndarray, np.ndarray]:
        nw, nd = x0.shape
        xnew = self.get_updated_vector(rng, x0)
        if self.mode == "random":
            m = (range(nw), rng.integers(x0.shape[-1], size=nw))
        elif self.mode == "sequential":
            m = (range(nw), self.index % nd + np.zeros(nw, dtype=int))
            self.index = (self.index + 1) % nd
        else:
            return xnew, np.zeros(nw)
        x = np.array(x0)
        x[m] = xnew[m]
        return x, np.zeros(nw)


class _diagonal_proposal(_isotropic_proposal):
    def get_updated_vector(
        self, rng: np.random.Generator, x0: np.ndarray
    ) -> np.ndarray:
        return x0 + self.get_factor(rng) * self.scale * rng.standard_normal(
            x0.shape
        )


class _proposal(_isotropic_proposal):
    allowed_modes = ["vector"]

    def get_updated_vector(
        self, rng: np.random.Generator, x0: np.ndarray
    ) -> np.ndarray:
        return x0 + self.get_factor(rng) * rng.multivariate_normal(
            np.zeros(len(self.scale)), self.scale, size=x0.shape[0]
        )
