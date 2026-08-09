from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ..state import State
from .move import Move

if TYPE_CHECKING:
    from collections.abc import Callable

    from ..model import Model

__all__ = ["MHMove"]


class MHMove(Move):
    r"""A general Metropolis-Hastings proposal

    Concrete implementations can be made by providing a ``proposal_function``
    argument that implements the proposal as described below.
    For standard Gaussian Metropolis moves, :class:`moves.GaussianMove` can be
    used.

    Args:
        proposal_function: The proposal function. It should take 2 arguments: a
            ``numpy.random.Generator`` instance and a ``(K, ndim)`` list
            of coordinate vectors. This function should return the proposed
            position and the log-ratio of the proposal probabilities
            (:math:`\ln q(x;\,x^\prime) - \ln q(x^\prime;\,x)` where
            :math:`x^\prime` is the proposed coordinate).
        ndim (Optional[int]): If this proposal is only valid for a specific
            dimension of parameter space, set that here.

    """

    ndim: int | None
    get_proposal: Callable[
        [np.ndarray, np.random.Generator], tuple[np.ndarray, np.ndarray]
    ]

    def __init__(
        self,
        proposal_function: Callable[
            [np.ndarray, np.random.Generator], tuple[np.ndarray, np.ndarray]
        ],
        ndim: int | None = None,
    ) -> None:
        self.ndim = ndim
        self.get_proposal = proposal_function

    def propose(self, model: Model, state: State) -> tuple[State, np.ndarray]:
        """Use the move to generate a proposal and compute the acceptance

        Args:
            model (Model): The model functions and random number generator
                used to compute the proposal.
            state (State): The current state of the ensemble.

        Returns:
            A tuple of the updated :class:`State` and a vector of booleans
            indicating which walkers were accepted.

        """
        # Check to make sure that the dimensions match.
        nwalkers, ndim = state.coords.shape
        if self.ndim is not None and self.ndim != ndim:
            raise ValueError("Dimension mismatch in proposal")
        if state.log_prob is None:
            raise ValueError(
                "a state with computed log probabilities is required "
                "to generate a proposal"
            )

        # Get the move-specific proposal.
        q, factors = self.get_proposal(state.coords, model.random)

        # Compute the lnprobs of the proposed position.
        new_log_probs, new_blobs = model.compute_log_prob_fn(q)

        # Loop over the walkers and update them accordingly.
        lnpdiff = new_log_probs - state.log_prob + factors
        accepted = np.log(model.random.random(nwalkers)) < lnpdiff

        # Update the parameters
        new_state = State(q, log_prob=new_log_probs, blobs=new_blobs)
        state = self.update(state, new_state, accepted)

        return state, accepted
