from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from ..model import Model
    from ..state import State

__all__ = ["Move"]


class Move:
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
        raise NotImplementedError(
            "The proposal must be implemented by subclasses"
        )

    def tune(self, state: State, accepted: np.ndarray) -> None:
        pass

    def update(
        self,
        old_state: State,
        new_state: State,
        accepted: np.ndarray,
        subset: np.ndarray | None = None,
    ) -> State:
        """Update a given subset of the ensemble with an accepted proposal

        Args:
            old_state (State): The state of the ensemble before the proposal.
                It is updated in place with the accepted proposals.
            new_state (State): The proposed state for the walkers selected by
                ``subset``.
            accepted: A vector of booleans, over the full ensemble, indicating
                which walkers were accepted.
            subset (Optional): A boolean mask indicating which walkers were
                included in the subset. This can be used, for example, when
                updating only the primary ensemble in a :class:`RedBlueMove`.

        Returns:
            State: The updated ensemble state (the same object as
            ``old_state``).

        """
        if old_state.log_prob is None or new_state.log_prob is None:
            raise ValueError(
                "states with computed log probabilities are required "
                "to update the ensemble"
            )

        if subset is None:
            subset = np.ones(len(old_state.coords), dtype=bool)
        m1 = subset & accepted
        m2 = accepted[subset]
        old_state.coords[m1] = new_state.coords[m2]
        old_state.log_prob[m1] = new_state.log_prob[m2]

        if new_state.blobs is not None:
            if old_state.blobs is None:
                raise ValueError(
                    "If you start sampling with a given log_prob, "
                    "you also need to provide the current list of "
                    "blobs at that position."
                )
            old_state.blobs[m1] = new_state.blobs[m2]

        return old_state
