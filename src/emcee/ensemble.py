from __future__ import annotations

import warnings
from collections.abc import Iterable, Mapping
from itertools import count
from typing import TYPE_CHECKING, Any

import numpy as np

from .backends import Backend
from .model import Model
from .moves import Move, StretchMove
from .pbar import get_progress_bar
from .state import State
from .utils import (
    deprecation_warning,
    ensure_rng,
    stored_state_to_generator,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Generator, Sequence

    from numpy.typing import ArrayLike, DTypeLike

__all__ = ["EnsembleSampler", "walkers_independent"]

try:
    # Try to import from numpy.exceptions (available in NumPy 1.25 and later)
    from numpy.exceptions import VisibleDeprecationWarning
except ImportError:
    # Fallback to the top-level numpy import (for older versions)
    from numpy import (
        VisibleDeprecationWarning,  # ty: ignore[unresolved-import]
    )


class EnsembleSampler:
    """An ensemble MCMC sampler

    If you are upgrading from an earlier version of emcee, you might notice
    that some arguments have been removed. The parameters that control the
    proposals have been moved to the :ref:`moves-user` interface (``a`` and
    ``live_dangerously``), and the parameters related to parallelization can
    now be controlled via the ``pool`` argument (:ref:`parallel`).

    Args:
        nwalkers (int): The number of walkers in the ensemble.
        ndim (int): Number of dimensions in the parameter space.
        log_prob_fn (callable): A function that takes a vector in the
            parameter space as input and returns the natural logarithm of the
            posterior probability (up to an additive constant) for that
            position.
        moves (Optional): This can be a single move object, a list of moves,
            or a "weighted" list of the form ``[(emcee.moves.StretchMove(),
            0.1), ...]``. When running, the sampler will randomly select a
            move from this list (optionally with weights) for each proposal.
            (default: :class:`StretchMove`)
        args (Optional): A list of extra positional arguments for
            ``log_prob_fn``. ``log_prob_fn`` will be called with the sequence
            ``log_prob_fn(p, *args, **kwargs)``.
        kwargs (Optional): A dict of extra keyword arguments for
            ``log_prob_fn``. ``log_prob_fn`` will be called with the sequence
            ``log_prob_fn(p, *args, **kwargs)``.
        pool (Optional): An object with a ``map`` method that follows the same
            calling sequence as the built-in ``map`` function. This is
            generally used to compute the log-probabilities for the ensemble
            in parallel.
        backend (Optional): Either a :class:`backends.Backend` or a subclass
            (like :class:`backends.HDFBackend`) that is used to store and
            serialize the state of the chain. By default, the chain is stored
            as a set of numpy arrays in memory, but new backends can be
            written to support other mediums.
        vectorize (Optional[bool]): If ``True``, ``log_prob_fn`` is expected
            to accept a list of position vectors instead of just one. Note
            that ``pool`` will be ignored if this is ``True``.
            (default: ``False``)
        parameter_names (Optional[Union[List[str], Dict[str, List[int]]]]):
            names of individual parameters or groups of parameters. If
            specified, the ``log_prob_fn`` will recieve a dictionary of
            parameters, rather than a ``np.ndarray``.
        rng (Optional): A source of randomness for reproducible sampling:
            either ``None`` (a fresh unseeded generator; default), an
            integer or ``numpy.random.SeedSequence`` seed, or a
            ``numpy.random.Generator`` or ``numpy.random.BitGenerator``
            (used as-is). A legacy ``numpy.random.RandomState`` is also
            accepted but deprecated. When ``rng`` is ``None`` and the
            backend already stores a random state, that state is restored
            instead; an explicit ``rng`` takes precedence over the backend.

    """

    _moves: Sequence[Move]
    _weights: np.ndarray
    pool: Any
    vectorize: bool
    blobs_dtype: DTypeLike | None
    ndim: int
    nwalkers: int
    backend: Backend
    _previous_state: State | None
    _random: np.random.Generator
    log_prob_fn: _FunctionWrapper
    params_are_named: bool
    parameter_names: Mapping[str, int | list[int]]

    def __init__(
        self,
        nwalkers: int,
        ndim: int,
        log_prob_fn: Callable[..., Any],
        pool: Any = None,
        moves: Move | Iterable[Move] | Iterable[tuple[Move, float]] | None = (
            None
        ),
        args: Iterable[Any] | None = None,
        kwargs: dict[str, Any] | None = None,
        backend: Backend | None = None,
        vectorize: bool = False,
        blobs_dtype: DTypeLike | None = None,
        parameter_names: dict[str, int | list[int]] | list[str] | None = None,
        rng: int
        | Sequence[int]
        | np.random.SeedSequence
        | np.random.BitGenerator
        | np.random.Generator
        | np.random.RandomState
        | None = None,
    ) -> None:
        # Parse the move schedule
        all_moves: Sequence[Move]
        weights: Any
        if moves is None:
            all_moves = [StretchMove()]
            weights = [1.0]
        elif isinstance(moves, Iterable):
            # Materialize first so that a one-shot iterator is not
            # exhausted by the ``zip`` probe below
            move_list = list(moves)
            try:
                all_moves, weights = zip(*move_list, strict=True)
            except TypeError:
                # The TypeError from ``zip`` proves that ``moves`` holds
                # plain moves, not ``(move, weight)`` pairs, which the
                # checker cannot infer from the union
                all_moves = move_list  # ty: ignore[invalid-assignment]
                weights = np.ones(len(move_list))
        else:
            all_moves = [moves]
            weights = [1.0]
        self._moves = all_moves
        self._weights = np.atleast_1d(weights).astype(float)
        self._weights /= np.sum(self._weights)

        if vectorize and pool is not None:
            warnings.warn(
                "The 'pool' argument is ignored when 'vectorize' is True",
                RuntimeWarning,
                stacklevel=2,
            )
        self.pool = pool
        self.vectorize = vectorize
        self.blobs_dtype = blobs_dtype

        self.ndim = ndim
        self.nwalkers = nwalkers
        self.backend = Backend() if backend is None else backend

        # Deal with re-used backends
        if not self.backend.initialized:
            self._previous_state = None
            self.reset()
            state = None
        else:
            # Check the backend shape
            if self.backend.shape != (self.nwalkers, self.ndim):
                raise ValueError(
                    f"the shape of the backend ({self.backend.shape}) is "
                    "incompatible with the shape of the sampler "
                    f"({(self.nwalkers, self.ndim)})"
                )

            # Get the last random state
            state = self.backend.random_state

            # Grab the last step so that we can restart
            it = self.backend.iteration
            if it > 0:
                self._previous_state = self.get_last_sample()

        # The sampler's random number generator; an explicit ``rng``
        # argument takes precedence over a state stored in the backend
        if rng is not None or state is None:
            self._random = ensure_rng(rng)
        else:
            try:
                self._random = stored_state_to_generator(state)
            except (TypeError, ValueError, IndexError, KeyError):
                warnings.warn(
                    "The random state stored in the backend could not be "
                    "restored; sampling will continue with a fresh "
                    "generator",
                    RuntimeWarning,
                    stacklevel=2,
                )
                self._random = ensure_rng(rng)
                # Don't re-apply the broken state when sampling resumes
                # from the last stored sample
                previous_state = getattr(self, "_previous_state", None)
                if previous_state is not None:
                    previous_state.random_state = None

        # Do a little bit of _magic_ to make the likelihood call with
        # ``args`` and ``kwargs`` pickleable.
        self.log_prob_fn = _FunctionWrapper(log_prob_fn, args, kwargs)

        # Save the parameter names
        self.params_are_named = parameter_names is not None
        if parameter_names is not None:
            if not isinstance(parameter_names, (list, dict)):
                raise TypeError(
                    "'parameter_names' must be a list or dict, got "
                    f"{type(parameter_names).__name__}"
                )

            # Don't support vectorizing yet
            if self.vectorize:
                raise ValueError(
                    "named parameters with vectorization unsupported for now"
                )

            # Check for duplicate names
            dupes = set()
            uniq = []
            for name in parameter_names:
                if name not in dupes:
                    uniq.append(name)
                    dupes.add(name)
            if len(uniq) != len(parameter_names):
                raise ValueError(f"duplicate parameters: {dupes}")

            named_parameters: dict[str, int | list[int]]
            if isinstance(parameter_names, list):
                # Check for all named
                if len(parameter_names) != ndim:
                    raise ValueError(
                        "name all parameters or set `parameter_names` to "
                        "`None`"
                    )
                # Convert a list to a dict
                named_parameters = {
                    name: i for i, name in enumerate(parameter_names)
                }
            else:
                named_parameters = parameter_names

            # Check not too many names
            if len(named_parameters) > ndim:
                raise ValueError("too many names")

            # Check all indices appear
            values = [
                v if isinstance(v, list) else [v]
                for v in named_parameters.values()
            ]
            flat_values = {item for sublist in values for item in sublist}
            if flat_values != set(np.arange(ndim)):
                raise ValueError(
                    f"not all values appear -- set should be 0 to {ndim - 1}"
                )
            self.parameter_names = named_parameters

    @property
    def random_state(self) -> Mapping[str, Any]:
        """
        The state of the internal random number generator: the
        ``bit_generator.state`` dict of a ``numpy.random.Generator``.
        Setting this property to ``None`` is a no-op; setting it to an
        invalid state raises a ``RuntimeWarning`` and leaves the generator
        unchanged. For backwards compatibility, a legacy
        ``RandomState.get_state()`` tuple is also accepted when setting.

        """
        return self._random.bit_generator.state

    @random_state.setter  # NOQA
    def random_state(self, state: Any) -> None:
        """
        Set the state of the internal random number generator. ``None`` is
        ignored; an invalid state raises a ``RuntimeWarning`` and leaves the
        generator unchanged.

        """
        if state is None:
            return
        try:
            self._random = stored_state_to_generator(state)
        except (TypeError, ValueError, IndexError, KeyError):
            warnings.warn(
                "Invalid random state ignored; the sampler's random number "
                "generator was left unchanged",
                RuntimeWarning,
                stacklevel=2,
            )

    @property
    def iteration(self) -> int:
        return self.backend.iteration

    def reset(self) -> None:
        """
        Reset the bookkeeping parameters

        """
        self.backend.reset(self.nwalkers, self.ndim)

    def __getstate__(self) -> dict[str, Any]:
        # In order to be generally picklable, we need to discard the pool
        # object before trying.
        d = self.__dict__.copy()
        d["pool"] = None
        return d

    def sample(
        self,
        initial_state: State | ArrayLike,
        iterations: int | None = 1,
        tune: bool = False,
        skip_initial_state_check: bool = False,
        thin_by: int = 1,
        store: bool = True,
        progress: bool | str = False,
        progress_kwargs: dict[str, Any] | None = None,
    ) -> Generator[State, None, None]:
        """Advance the chain as a generator

        Args:
            initial_state (State or ndarray[nwalkers, ndim]): The initial
                :class:`State` or positions of the walkers in the
                parameter space.
            iterations (Optional[int or NoneType]): The number of steps to
                generate. ``None`` generates an infinite stream (requires
                ``store=False``).
            tune (Optional[bool]): If ``True``, the parameters of some moves
                will be automatically tuned.
            thin_by (Optional[int]): If you only want to store and yield every
                ``thin_by`` samples in the chain, set ``thin_by`` to an
                integer greater than 1. When this is set, ``iterations *
                thin_by`` proposals will be made.
            store (Optional[bool]): By default, the sampler stores (in memory)
                the positions and log-probabilities of the samples in the
                chain. If you are using another method to store the samples to
                a file or if you don't need to analyze the samples after the
                fact (for burn-in for example) set ``store`` to ``False``.
            progress (Optional[bool or str]): If ``True``, a progress bar will
                be shown as the sampler progresses. If a string, will select a
                specific ``tqdm`` progress bar - most notable is
                ``'notebook'``, which shows a progress bar suitable for
                Jupyter notebooks.  If ``False``, no progress bar will be
                shown.
            progress_kwargs (Optional[dict]): A ``dict`` of keyword arguments
                to be passed to the tqdm call.
            skip_initial_state_check (Optional[bool]): If ``True``, a check
                that the initial_state can fully explore the space will be
                skipped. (default: ``False``)


        Every ``thin_by`` steps, this generator yields the
        :class:`State` of the ensemble.

        """
        if store and iterations is None:
            raise ValueError("'store' must be False when 'iterations' is None")
        # Interpret the input as a walker state and check the dimensions.
        state = State(initial_state, copy=True)
        state_shape = np.shape(state.coords)
        if state_shape != (self.nwalkers, self.ndim):
            raise ValueError(f"incompatible input dimensions {state_shape}")
        if (not skip_initial_state_check) and (
            not walkers_independent(state.coords)
        ):
            raise ValueError(
                "Initial state has a large condition number. "
                "Make sure that your walkers are linearly independent for the "
                "best performance"
            )

        # Set the initial value of the random number generator. A state of
        # ``None`` (e.g. when ``initial_state`` was a plain array) leaves the
        # generator in its current state.
        self.random_state = state.random_state

        # If the initial log-probabilities were not provided, calculate them
        # now.
        if state.log_prob is None:
            state.log_prob, state.blobs = self.compute_log_prob(state.coords)
        if np.shape(state.log_prob) != (self.nwalkers,):
            raise ValueError("incompatible input dimensions")

        # Check to make sure that the probability function didn't return
        # ``np.nan``.
        if np.any(np.isnan(state.log_prob)):
            raise ValueError("The initial log_prob was NaN")

        # Check that the thin keyword is reasonable.
        thin_by = int(thin_by)
        if thin_by <= 0:
            raise ValueError("Invalid thinning argument")

        yield_step = thin_by
        checkpoint_step = thin_by
        if store and iterations is not None:
            self.backend.grow(iterations, state.blobs)

        # Set up a wrapper around the relevant model functions
        if self.pool is not None:
            map_fn = self.pool.map
        else:
            map_fn = map
        if progress_kwargs is None:
            progress_kwargs = {}

        # Inject the progress bar
        total = None if iterations is None else iterations * yield_step
        with get_progress_bar(progress, total, **progress_kwargs) as pbar:
            i = 0
            for _ in count() if iterations is None else range(iterations):
                for _ in range(yield_step):
                    # Rebuild the model wrapper every step so that a
                    # generator swapped in through the ``random_state``
                    # setter mid-run is picked up by the moves
                    model = Model(
                        self.log_prob_fn,
                        self.compute_log_prob,
                        map_fn,
                        self._random,
                    )

                    # Choose a random move
                    move = self._moves[
                        self._random.choice(len(self._moves), p=self._weights)
                    ]

                    # Propose
                    state, accepted = move.propose(model, state)
                    state.random_state = self.random_state

                    if tune:
                        move.tune(state, accepted)

                    # Save the new step
                    if store and (i + 1) % checkpoint_step == 0:
                        self.backend.save_step(state, accepted)

                    pbar.update(1)
                    i += 1

                # Yield the result as an iterator so that the user can do all
                # sorts of fun stuff with the results so far.
                yield state

    def run_mcmc(
        self,
        initial_state: State | ArrayLike | None,
        nsteps: int,
        **kwargs: Any,
    ) -> State | None:
        """
        Iterate :func:`sample` for ``nsteps`` iterations and return the result

        Args:
            initial_state: The initial state or position vector. Can also be
                ``None`` to resume from where :func:``run_mcmc`` left off the
                last time it executed.
            nsteps: The number of steps to run.

        Other parameters are directly passed to :func:`sample`.

        This method returns the most recent result from :func:`sample`.

        """
        if initial_state is None:
            if self._previous_state is None:
                raise ValueError(
                    "Cannot have `initial_state=None` if run_mcmc has never "
                    "been called."
                )
            initial_state = self._previous_state

        results = None
        # The loop variable is read after the loop: only the final state of
        # the chain is kept.
        for results in self.sample(  # noqa: B007
            initial_state, iterations=nsteps, **kwargs
        ):
            pass

        # Store so that the ``initial_state=None`` case will work
        self._previous_state = results

        return results

    def compute_log_prob(
        self, coords: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """Calculate the vector of log-probability for the walkers

        Args:
            coords: (ndarray[..., ndim]) The position vector in parameter
                space where the probability should be calculated.

        This method returns:

        * log_prob: A vector of log-probabilities with one entry for each
          walker in this sub-ensemble.
        * blob: The list of meta data returned by the ``log_post_fn`` at
          this position or ``None`` if nothing was returned.

        """
        p = coords

        # Check that the parameters are in physical ranges.
        if np.any(np.isinf(p)):
            raise ValueError("At least one parameter value was infinite")
        if np.any(np.isnan(p)):
            raise ValueError("At least one parameter value was NaN")

        # If the parmaeters are named, then switch to dictionaries
        if self.params_are_named:
            p = ndarray_to_list_of_dicts(p, self.parameter_names)

        # Run the log-probability calculations (optionally in parallel).
        if self.vectorize:
            results = self.log_prob_fn(p)
        else:
            # If the `pool` property of the sampler has been set (i.e. we want
            # to use `multiprocessing`), use the `pool`'s map method.
            # Otherwise, just use the built-in `map` function.
            if self.pool is not None:
                map_func = self.pool.map
            else:
                map_func = map
            results = list(map_func(self.log_prob_fn, p))

        # Does the log-prob function return blobs (extra values beyond the
        # log-probability)? A bare scalar or a length-1 sequence (e.g.
        # ``np.array([1.234])``) means no blobs.
        try:
            lengths = [len(res) for res in results]
            has_blobs = any(n > 1 for n in lengths)
        except TypeError:
            has_blobs = False

        if not has_blobs:
            log_prob = np.array([_scalar(res) for res in results])
            blob = None
        else:
            if any(n <= 1 for n in lengths):
                raise ValueError(
                    "The log probability function returned blobs for some "
                    "walkers but not others"
                )
            blob = [res[1:] for res in results]
            log_prob = np.array([_scalar(res[0]) for res in results])

            # Get the blobs dtype
            if self.blobs_dtype is not None:
                dt = self.blobs_dtype
            else:
                try:
                    with warnings.catch_warnings(record=True):
                        warnings.simplefilter(
                            "error", VisibleDeprecationWarning
                        )
                        try:
                            dt = np.atleast_1d(blob[0]).dtype
                        except Warning:
                            deprecation_warning(
                                "You have provided blobs that are not all the "
                                "same shape or size. This means they must be "
                                "placed in an object array. Numpy has "
                                "deprecated this automatic detection, so "
                                "please specify "
                                "blobs_dtype=np.dtype('object')"
                            )
                            dt = np.dtype("object")
                except ValueError:
                    dt = np.dtype("object")
                if dt.kind in "US":
                    # Strings need to be object arrays or we risk truncation
                    dt = np.dtype("object")
            blob = np.array(blob, dtype=dt)

            # Deal with single blobs properly
            shape = blob.shape[1:]
            if len(shape):
                axes = np.arange(len(shape))[np.array(shape) == 1] + 1
                if len(axes):
                    blob = np.squeeze(blob, tuple(axes))

        # Check for log_prob returning NaN.
        if np.any(np.isnan(log_prob)):
            raise ValueError("Probability function returned NaN")

        return log_prob, blob

    @property
    def acceptance_fraction(self) -> np.ndarray:
        """The fraction of proposed steps that were accepted"""
        return self.backend.accepted / float(self.backend.iteration)

    def get_chain(self, **kwargs: Any) -> np.ndarray:
        return self.get_value("chain", **kwargs)

    get_chain.__doc__ = Backend.get_chain.__doc__

    def get_blobs(self, **kwargs: Any) -> np.ndarray | None:
        return self.get_value("blobs", **kwargs)

    get_blobs.__doc__ = Backend.get_blobs.__doc__

    def get_log_prob(self, **kwargs: Any) -> np.ndarray:
        return self.get_value("log_prob", **kwargs)

    get_log_prob.__doc__ = Backend.get_log_prob.__doc__

    def get_last_sample(self, **kwargs: Any) -> State:
        return self.backend.get_last_sample()

    get_last_sample.__doc__ = Backend.get_last_sample.__doc__

    def get_value(self, name: str, **kwargs: Any) -> Any:
        return self.backend.get_value(name, **kwargs)

    def get_autocorr_time(self, **kwargs: Any) -> np.ndarray:
        return self.backend.get_autocorr_time(**kwargs)

    get_autocorr_time.__doc__ = Backend.get_autocorr_time.__doc__


class _FunctionWrapper:
    """
    This is a hack to make the likelihood function pickleable when ``args``
    or ``kwargs`` are also included.

    """

    f: Callable[..., Any]
    args: Sequence[Any]
    kwargs: dict[str, Any]

    def __init__(
        self,
        f: Callable[..., Any],
        args: Iterable[Any] | None,
        kwargs: dict[str, Any] | None,
    ) -> None:
        self.f = f
        self.args = list(args) if args is not None else []
        self.kwargs = kwargs or {}

    def __call__(self, x: Any) -> Any:
        try:
            return self.f(x, *self.args, **self.kwargs)
        except Exception:  # pragma: no cover
            import traceback

            print("emcee: Exception while calling your likelihood function:")
            print("  params:", x)
            print("  args:", self.args)
            print("  kwargs:", self.kwargs)
            print("  exception:")
            traceback.print_exc()
            raise


def walkers_independent(coords: ArrayLike) -> bool:
    coords = np.asarray(coords)
    if not np.all(np.isfinite(coords)):
        return False
    C = coords - np.mean(coords, axis=0)[None, :]
    C_colmax = np.amax(np.abs(C), axis=0)
    if np.any(C_colmax == 0):
        return False
    C /= C_colmax
    C_colsum = np.sqrt(np.sum(C**2, axis=0))
    C /= C_colsum
    return bool(np.linalg.cond(C.astype(float)) <= 1e8)


def ndarray_to_list_of_dicts(
    x: np.ndarray, key_map: Mapping[str, int | list[int]]
) -> list[dict[str, np.number | np.ndarray]]:
    """
    A helper function to convert a ``np.ndarray`` into a list
    of dictionaries of parameters. Used when parameters are named.

    Args:
      x (np.ndarray): parameter array of shape ``(N, n_dim)``, where
        ``N`` is an integer
      key_map (Dict[str, Union[int, List[int]]):

    Returns:
      list of dictionaries of parameters
    """
    return [{key: xi[val] for key, val in key_map.items()} for xi in x]


def _scalar(fx: Any) -> float:
    # Make sure a value is a true scalar
    # 1.0, np.float64(1.0), np.array([1.0]), np.array(1.0)
    if not np.isscalar(fx):
        try:
            fx = np.asarray(fx).item()
        except (TypeError, ValueError) as e:
            raise ValueError("log_prob_fn should return scalar") from e
        return float(fx)
    else:
        # ``np.isscalar`` narrows ``fx`` to a union that includes
        # ``complex``; a complex log-probability raising TypeError at
        # runtime is the desired behavior
        return float(fx)  # ty: ignore[invalid-argument-type]
