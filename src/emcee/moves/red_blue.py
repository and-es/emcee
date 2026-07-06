import numpy as np

from ..state import State
from .move import Move

__all__ = ["RedBlueMove"]


class RedBlueMove(Move):
    """
    An abstract red-blue ensemble move with parallelization as described in
    `Foreman-Mackey et al. (2013) <https://arxiv.org/abs/1202.3665>`_.

    Args:
        nsplits (Optional[int]): The number of sub-ensembles to use. Each
            sub-ensemble is updated in parallel using the other sets as the
            complementary ensemble. The default value is ``2`` and you
            probably won't need to change that.

        randomize_split (Optional[bool]): Randomly shuffle walkers between
            sub-ensembles. The same number of walkers will be assigned to
            each sub-ensemble on each iteration. By default, this is ``True``.

        live_dangerously (Optional[bool]): By default, an update will fail with
            a ``RuntimeError`` if the number of walkers is smaller than twice
            the dimension of the problem because the walkers would then be
            stuck on a low dimensional subspace. This can be avoided by
            switching between the stretch move and, for example, a
            Metropolis-Hastings step. If you want to do this and suppress the
            error, set ``live_dangerously = True``. Thanks goes (once again)
            to @dstndstn for this wonderful terminology.

    """

    def __init__(
        self, nsplits=2, randomize_split=True, live_dangerously=False
    ):
        self.nsplits = int(nsplits)
        self.live_dangerously = live_dangerously
        self.randomize_split = randomize_split

    def setup(self, coords):
        pass

    def get_proposal(self, sample, complement, random):
        """Generate a proposal for a sub-ensemble

        Args:
            sample: The coordinates of the walkers being updated.
            complement: A list of coordinate arrays, one per complementary
                sub-ensemble.
            random: A ``numpy.random.Generator`` instance.

        Returns:
            A tuple of the proposed coordinates and a vector of the
            log-ratios of the proposal probabilities.

        """
        raise NotImplementedError(
            "The proposal must be implemented by subclasses"
        )

    def propose(self, model, state):
        """Use the move to generate a proposal and compute the acceptance

        Args:
            model (Model): The model functions and random number generator
                used to compute the proposal.
            state (State): The current state of the ensemble.

        Returns:
            A tuple of the updated :class:`State` and a vector of booleans
            indicating which walkers were accepted.

        """
        # Check that the dimensions are compatible.
        nwalkers, ndim = state.coords.shape
        if nwalkers < 2 * ndim and not self.live_dangerously:
            raise RuntimeError(
                "It is unadvisable to use a red-blue move "
                "with fewer walkers than twice the number of "
                "dimensions."
            )

        # Run any move-specific setup.
        self.setup(state.coords)

        # Split the ensemble in half and iterate over these two halves.
        accepted = np.zeros(nwalkers, dtype=bool)
        inds = np.arange(nwalkers) % self.nsplits
        if self.randomize_split:
            model.random.shuffle(inds)
        # The masks are loop-invariant; the coordinate sets are not, because
        # each split updates ``state.coords`` in place.
        masks = [inds == j for j in range(self.nsplits)]
        for split in range(self.nsplits):
            S1 = masks[split]

            # Get the two halves of the ensemble.
            sets = [state.coords[m] for m in masks]
            s = sets[split]
            c = sets[:split] + sets[split + 1 :]

            # Get the move-specific proposal.
            q, factors = self.get_proposal(s, c, model.random)

            # Compute the lnprobs of the proposed position.
            new_log_probs, new_blobs = model.compute_log_prob_fn(q)

            # Decide the acceptance for each walker in the split.
            lnpdiff = factors + new_log_probs - state.log_prob[S1]
            accepted[S1] = lnpdiff > np.log(model.random.random(len(lnpdiff)))

            new_state = State(q, log_prob=new_log_probs, blobs=new_blobs)
            state = self.update(state, new_state, accepted, S1)

        return state, accepted
