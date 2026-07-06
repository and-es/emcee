import pickle
from itertools import islice, product

import numpy as np
import pytest

from emcee import EnsembleSampler, State, backends, moves, walkers_independent

try:
    import tqdm
except ImportError:
    tqdm = None  # ty: ignore[invalid-assignment]

__all__ = ["test_shapes", "test_errors", "test_vectorize"]

all_backends = backends.get_test_backends()


def normal_log_prob(params):
    return -0.5 * np.sum(params**2)


@pytest.mark.parametrize(
    "backend, moves",
    list(
        product(
            all_backends,
            [
                None,
                moves.GaussianMove(0.5),
                [moves.StretchMove(), moves.GaussianMove(0.5)],
                [
                    (moves.StretchMove(), 0.3),
                    (moves.GaussianMove(0.5), 0.1),
                ],
            ],
        )
    ),
)
def test_shapes(backend, moves, nwalkers=32, ndim=3, nsteps=10, seed=1234):
    # Set up the random number generator.
    rng = np.random.default_rng(seed)

    with backend() as be:
        # Initialize the ensemble, moves and sampler.
        coords = rng.standard_normal((nwalkers, ndim))
        sampler = EnsembleSampler(
            nwalkers, ndim, normal_log_prob, moves=moves, backend=be
        )

        # Run the sampler.
        sampler.run_mcmc(coords, nsteps)
        chain = sampler.get_chain()
        assert len(chain) == nsteps, "wrong number of steps"

        tau = sampler.get_autocorr_time(quiet=True)
        assert tau.shape == (ndim,)

        # Check the shapes.
        assert sampler.get_chain().shape == (
            nsteps,
            nwalkers,
            ndim,
        ), "incorrect coordinate dimensions"
        assert sampler.get_log_prob().shape == (
            nsteps,
            nwalkers,
        ), "incorrect probability dimensions"

        assert sampler.acceptance_fraction.shape == (nwalkers,), (
            "incorrect acceptance fraction dimensions"
        )

        # Check the shape of the flattened coords.
        assert sampler.get_chain(flat=True).shape == (
            nsteps * nwalkers,
            ndim,
        ), "incorrect coordinate dimensions"
        assert sampler.get_log_prob(flat=True).shape == (nsteps * nwalkers,), (
            "incorrect probability dimensions"
        )


@pytest.mark.parametrize("backend", all_backends)
def test_errors(backend, nwalkers=32, ndim=3, nsteps=5, seed=1234):
    # Set up the random number generator.
    rng = np.random.default_rng(seed)

    with backend() as be:
        # Initialize the ensemble, proposal, and sampler.
        coords = rng.standard_normal((nwalkers, ndim))
        sampler = EnsembleSampler(nwalkers, ndim, normal_log_prob, backend=be)

        # Test for not running.
        with pytest.raises(AttributeError):
            sampler.get_chain()
        with pytest.raises(AttributeError):
            sampler.get_log_prob()
        with pytest.raises(AttributeError):
            _ = sampler.acceptance_fraction

        # What about not storing the chain.
        sampler.run_mcmc(coords, nsteps, store=False)
        with pytest.raises(AttributeError):
            sampler.get_chain()

        # Now what about if we try to continue using the sampler with an
        # ensemble of a different shape.
        sampler.run_mcmc(coords, nsteps, store=False)

        coords2 = rng.standard_normal((nwalkers, ndim + 1))
        with pytest.raises(ValueError):
            sampler.run_mcmc(coords2, nsteps)

        # Ensure that a warning is logged if the inital coords don't allow
        # the chain to explore all of parameter space, and that one is not
        # if we explicitly disable it, or the initial coords can.
        with pytest.raises(ValueError):
            sampler.run_mcmc(np.ones((nwalkers, ndim)), nsteps)
        sampler.run_mcmc(
            np.ones((nwalkers, ndim)), nsteps, skip_initial_state_check=True
        )
        sampler.run_mcmc(rng.standard_normal((nwalkers, ndim)), nsteps)


def run_sampler(
    backend,
    nwalkers=32,
    ndim=3,
    nsteps=25,
    seed=1234,
    thin_by=1,
    progress=False,
    store=True,
):
    coords = np.random.default_rng(seed).standard_normal((nwalkers, ndim))
    sampler = EnsembleSampler(
        nwalkers, ndim, normal_log_prob, backend=backend, rng=seed
    )
    sampler.run_mcmc(
        coords,
        nsteps,
        thin_by=thin_by,
        progress=progress,
        store=store,
    )
    return sampler


@pytest.mark.parametrize(
    "backend,progress", list(product(all_backends, [True, False]))
)
def test_thin_by(backend, progress):
    with backend() as be:
        with pytest.raises(ValueError):
            run_sampler(be, thin_by=-1)
        with pytest.raises(ValueError):
            run_sampler(be, thin_by=0.1)
        nsteps = 25
        thinby = 3
        sampler1 = run_sampler(None, nsteps=nsteps * thinby, progress=progress)
        sampler2 = run_sampler(
            be, thin_by=thinby, progress=progress, nsteps=nsteps
        )
        for k in ["get_chain", "get_log_prob"]:
            a = getattr(sampler1, k)()[thinby - 1 :: thinby]
            b = getattr(sampler2, k)()
            c = getattr(sampler1, k)(thin=thinby)
            assert np.allclose(a, b), f"inconsistent {k}"
            assert np.allclose(a, c), f"inconsistent {k}"
        assert sampler1.iteration == sampler2.iteration * thinby


@pytest.mark.parametrize("backend", all_backends)
def test_restart(backend):
    with backend() as be:
        sampler = run_sampler(be, nsteps=0)
        with pytest.raises(ValueError):
            sampler.run_mcmc(None, 10)

        sampler = run_sampler(be)
        sampler.run_mcmc(None, 10)

    with backend() as be:
        sampler = run_sampler(be, store=False)
        sampler.run_mcmc(None, 10)


def test_vectorize():
    def lp_vec(p):
        return -0.5 * np.sum(p**2, axis=1)

    nwalkers, ndim = 32, 3
    coords = np.random.default_rng(42).standard_normal((nwalkers, ndim))
    sampler = EnsembleSampler(nwalkers, ndim, lp_vec, vectorize=True)
    sampler.run_mcmc(coords, 10)

    assert sampler.get_chain().shape == (10, nwalkers, ndim)


def test_vectorize_with_pool_warns(nwalkers=32, ndim=3):
    class FakePool:
        def map(self, fn, iterable):
            return map(fn, iterable)

    def lp_vec(p):
        return -0.5 * np.sum(p**2, axis=1)

    with pytest.warns(RuntimeWarning, match="'pool' argument is ignored"):
        EnsembleSampler(
            nwalkers, ndim, lp_vec, vectorize=True, pool=FakePool()
        )


@pytest.mark.parametrize("backend", all_backends)
def test_pickle(backend):
    with backend() as be:
        sampler1 = run_sampler(be)
        s = pickle.dumps(sampler1, -1)
        sampler2 = pickle.loads(s)
        for k in ["get_chain", "get_log_prob"]:
            a = getattr(sampler1, k)()
            b = getattr(sampler2, k)()
            assert np.allclose(a, b), f"inconsistent {k}"


def test_pickle_preserves_pool():
    # Pickling discards the pool from the copy, but it must not remove it
    # from the live sampler.
    class FakePool:
        def map(self, func, iterable):
            return map(func, iterable)

    pool = FakePool()
    sampler = EnsembleSampler(32, 3, normal_log_prob, pool=pool)
    pickled = pickle.loads(pickle.dumps(sampler, -1))
    assert sampler.pool is pool
    assert pickled.pool is None


@pytest.mark.parametrize("nwalkers, ndim", [(10, 2), (20, 5)])
def test_walkers_dependent_ones(nwalkers, ndim):
    assert not walkers_independent(np.ones((nwalkers, ndim)))


@pytest.mark.parametrize("nwalkers, ndim", [(10, 11), (2, 3)])
def test_walkers_dependent_toofew(nwalkers, ndim):
    rng = np.random.default_rng(8231)
    assert not walkers_independent(rng.standard_normal((nwalkers, ndim)))


@pytest.mark.parametrize("nwalkers, ndim", [(10, 2), (20, 5)])
def test_walkers_independent_randn(nwalkers, ndim):
    rng = np.random.default_rng(8231)
    assert walkers_independent(rng.standard_normal((nwalkers, ndim)))


@pytest.mark.parametrize(
    "nwalkers, ndim, offset", [(10, 2, 1e5), (20, 5, 1e10), (30, 10, 1e14)]
)
def test_walkers_independent_randn_offset(nwalkers, ndim, offset):
    rng = np.random.default_rng(8231)
    assert walkers_independent(
        rng.standard_normal((nwalkers, ndim))
        + np.ones((nwalkers, ndim)) * offset
    )


def test_walkers_dependent_big_offset():
    nwalkers, ndim = 30, 10
    rng = np.random.default_rng(8231)
    offset = 10 / np.finfo(float).eps
    assert not walkers_independent(
        rng.standard_normal((nwalkers, ndim))
        + np.ones((nwalkers, ndim)) * offset
    )


def test_walkers_dependent_subtle():
    nwalkers, ndim = 30, 10
    rng = np.random.default_rng(8231)
    w = rng.standard_normal((nwalkers, ndim))
    assert walkers_independent(w)
    # random unit vector
    p = rng.standard_normal(ndim)
    p /= np.sqrt(np.dot(p, p))
    # project away the direction of p
    w -= np.sum(p[None, :] * w, axis=1)[:, None] * p[None, :]
    assert not walkers_independent(w)
    # shift away from the origin
    w += p[None, :]
    assert not walkers_independent(w)


def test_walkers_almost_dependent():
    nwalkers, ndim = 30, 10
    rng = np.random.default_rng(8231)
    squash = 1e-8
    w = rng.standard_normal((nwalkers, ndim))
    assert walkers_independent(w)
    # random unit vector
    p = rng.standard_normal(ndim)
    p /= np.sqrt(np.dot(p, p))
    # project away the direction of p
    proj = np.sum(p[None, :] * w, axis=1)[:, None] * p[None, :]
    w -= proj
    w += squash * proj
    assert not walkers_independent(w)


def test_walkers_independent_scaled():
    # Some of these scales will overflow if squared, hee hee
    scales = np.array([1, 1e10, 1e100, 1e200, 1e-10, 1e-100, 1e-200])
    ndim = len(scales)
    nwalkers = 5 * ndim
    rng = np.random.default_rng(8231)
    w = rng.standard_normal((nwalkers, ndim)) * scales[None, :]
    assert walkers_independent(w)


@pytest.mark.parametrize(
    "nwalkers, ndim, offset",
    [
        (10, 2, 1e5),
        (20, 5, 1e10),
        (30, 10, 1e14),
        (40, 15, 0.1 / np.finfo(np.longdouble).eps),
    ],
)
def test_walkers_independent_randn_offset_longdouble(nwalkers, ndim, offset):
    rng = np.random.default_rng(8231)
    assert walkers_independent(
        rng.standard_normal((nwalkers, ndim))
        + np.ones((nwalkers, ndim), dtype=np.longdouble) * offset
    )


def test_pool_used_for_sampling():
    class CountingPool:
        def __init__(self):
            self.count = 0

        def map(self, func, iterable):
            self.count += 1
            return map(func, iterable)

    sampler1 = run_sampler(None)

    pool = CountingPool()
    coords = np.random.default_rng(1234).standard_normal((32, 3))
    sampler2 = EnsembleSampler(32, 3, normal_log_prob, pool=pool, rng=1234)
    sampler2.run_mcmc(coords, 25)

    assert pool.count > 0
    for k in ["get_chain", "get_log_prob"]:
        a = getattr(sampler1, k)()
        b = getattr(sampler2, k)()
        assert np.allclose(a, b), f"inconsistent {k}"


def test_tune(nwalkers=32, ndim=3, nsteps=5, seed=1234):
    coords = np.random.default_rng(seed).standard_normal((nwalkers, ndim))

    # The base move implements tune() as a no-op.
    sampler = EnsembleSampler(nwalkers, ndim, normal_log_prob)
    sampler.run_mcmc(coords, nsteps, tune=True)
    assert sampler.get_chain().shape == (nsteps, nwalkers, ndim)

    # A move that overrides tune() must be called once per step.
    class TuningMove(moves.StretchMove):
        ncalls = 0

        def tune(self, state, accepted):
            self.ncalls += 1

    move = TuningMove()
    sampler = EnsembleSampler(nwalkers, ndim, normal_log_prob, moves=move)
    sampler.run_mcmc(coords, nsteps, tune=True)
    assert move.ncalls == nsteps


@pytest.mark.skipif(tqdm is None, reason="tqdm not available")
def test_progress_kwargs(capsys, nwalkers=32, ndim=3, seed=1234):
    coords = np.random.default_rng(seed).standard_normal((nwalkers, ndim))
    sampler = EnsembleSampler(nwalkers, ndim, normal_log_prob)
    sampler.run_mcmc(
        coords, 5, progress=True, progress_kwargs={"desc": "emcee-test"}
    )
    assert "emcee-test" in capsys.readouterr().err


def test_incompatible_backend_shape():
    be = backends.Backend()
    run_sampler(be)
    with pytest.raises(ValueError, match="incompatible"):
        EnsembleSampler(10, 2, normal_log_prob, backend=be)


def test_nan_initial_log_prob(nwalkers=32, ndim=3, seed=1234):
    coords = np.random.default_rng(seed).standard_normal((nwalkers, ndim))
    state = State(coords, log_prob=np.full(nwalkers, np.nan))
    sampler = EnsembleSampler(nwalkers, ndim, normal_log_prob)
    with pytest.raises(ValueError, match="initial log_prob was NaN"):
        sampler.run_mcmc(state, 1)


def test_log_prob_fn_returns_nan(nwalkers=32, ndim=3, seed=1234):
    coords = np.random.default_rng(seed).standard_normal((nwalkers, ndim))
    sampler = EnsembleSampler(nwalkers, ndim, lambda p: np.nan)
    with pytest.raises(ValueError, match="returned NaN"):
        sampler.run_mcmc(coords, 1)


def test_log_prob_fn_returns_non_scalar(nwalkers=32, ndim=3, seed=1234):
    coords = np.random.default_rng(seed).standard_normal((nwalkers, ndim))
    sampler = EnsembleSampler(nwalkers, ndim, lambda p: np.zeros((1, 2)))
    with pytest.raises(ValueError, match="should return scalar"):
        sampler.run_mcmc(coords, 1)


def test_random_state_setter(nwalkers=32, ndim=3):
    sampler = EnsembleSampler(nwalkers, ndim, normal_log_prob)
    state = sampler.random_state
    assert isinstance(state, dict)

    # Setting ``None`` is a silent no-op.
    sampler.random_state = None
    assert sampler.random_state == state

    # An invalid state warns and leaves the generator unchanged.
    garbage_states = [
        (),
        "garbage",
        ("MT19937",),
        42,
        {},
        {"bit_generator": "NotABitGenerator"},
    ]
    for garbage in garbage_states:
        with pytest.warns(RuntimeWarning, match="Invalid random state"):
            sampler.random_state = garbage
        assert sampler.random_state == state

    # A state dict for the same bit generator is applied in place.
    other = np.random.default_rng(42).bit_generator.state
    sampler.random_state = other
    assert sampler.random_state == other

    # A state dict for a different bit generator replaces the generator.
    mt_state = np.random.Generator(np.random.MT19937(1)).bit_generator.state
    sampler.random_state = mt_state
    new = sampler.random_state
    assert new["bit_generator"] == "MT19937"
    assert np.array_equal(new["state"]["key"], mt_state["state"]["key"])
    assert new["state"]["pos"] == mt_state["state"]["pos"]

    # A legacy RandomState tuple is converted to an MT19937 generator.
    legacy = np.random.mtrand.RandomState(42).get_state()
    sampler.random_state = legacy
    new = sampler.random_state
    assert new["bit_generator"] == "MT19937"
    # numpy stubs type get_state() as a dict, but legacy=True returns
    # a tuple
    assert np.array_equal(
        new["state"]["key"],
        legacy[1],  # ty: ignore[invalid-argument-type]
    )
    assert new["state"]["pos"] == legacy[2]  # ty: ignore[invalid-argument-type]


def test_random_state_set_mid_run(nwalkers=32, ndim=3):
    # A state set through the setter between two steps of a running
    # ``sample()`` generator must drive the following proposals.
    coords = np.random.default_rng(0).standard_normal((nwalkers, ndim))
    mt_state = np.random.Generator(np.random.MT19937(7)).bit_generator.state

    sampler1 = EnsembleSampler(nwalkers, ndim, normal_log_prob, rng=100)
    gen = sampler1.sample(coords, iterations=2, store=False)
    mid = next(gen)
    sampler1.random_state = mt_state
    final1 = next(gen)

    # A fresh sampler continuing from the same midpoint with the same
    # state must produce the same step. The yielded midpoint is a
    # snapshot, so it is still valid after the second step above.
    sampler2 = EnsembleSampler(nwalkers, ndim, normal_log_prob, rng=200)
    sampler2.random_state = mt_state
    final2 = sampler2.run_mcmc(
        State(mid.coords, log_prob=mid.log_prob), 1, store=False
    )

    assert final2 is not None
    assert np.allclose(final1.coords, final2.coords)


def test_yielded_states_are_snapshots(nwalkers=32, ndim=3):
    # Each yielded State has its own arrays and is not mutated by later
    # steps, so collecting them tracks the stored chain
    coords = np.random.default_rng(456).standard_normal((nwalkers, ndim))
    sampler = EnsembleSampler(nwalkers, ndim, normal_log_prob, rng=9)

    states = list(sampler.sample(coords, iterations=3))
    assert len({id(s) for s in states}) == 3
    assert np.array_equal(
        np.stack([s.coords for s in states]), sampler.get_chain()
    )
    assert np.array_equal(
        np.stack([s.log_prob for s in states]), sampler.get_log_prob()
    )

    # With thinning, the yields line up with the stored steps
    sampler.reset()
    states = list(sampler.sample(coords, iterations=3, thin_by=2))
    assert len({id(s) for s in states}) == 3
    assert np.array_equal(
        np.stack([s.coords for s in states]), sampler.get_chain()
    )


def test_sample_copy_false(nwalkers=32, ndim=3):
    # copy=False opts back into the historical behavior: the same State
    # object is updated in place and yielded every time
    coords = np.random.default_rng(456).standard_normal((nwalkers, ndim))
    sampler = EnsembleSampler(nwalkers, ndim, normal_log_prob, rng=9)
    states = list(
        sampler.sample(coords, iterations=3, store=False, copy=False)
    )
    assert all(s is states[0] for s in states)


def test_rng_reproducibility(nwalkers=32, ndim=3):
    coords = np.random.default_rng(1234).standard_normal((nwalkers, ndim))

    def run(rng):
        sampler = EnsembleSampler(nwalkers, ndim, normal_log_prob, rng=rng)
        sampler.run_mcmc(coords, 10)
        return sampler.get_chain()

    assert np.array_equal(run(42), run(42))
    assert not np.array_equal(run(42), run(43))


def test_rng_argument(nwalkers=32, ndim=3):
    # A ``Generator`` is used as-is.
    gen = np.random.default_rng(42)
    sampler = EnsembleSampler(nwalkers, ndim, normal_log_prob, rng=gen)
    assert (
        sampler.random_state == np.random.default_rng(42).bit_generator.state
    )

    # A legacy ``RandomState`` is deprecated but converted.
    with pytest.warns(DeprecationWarning, match="RandomState"):
        sampler = EnsembleSampler(
            nwalkers, ndim, normal_log_prob, rng=np.random.RandomState(42)
        )
    assert sampler.random_state["bit_generator"] == "MT19937"

    # Invalid seeds raise a ``TypeError``.
    with pytest.raises(TypeError):
        EnsembleSampler(
            nwalkers,
            ndim,
            normal_log_prob,
            rng="invalid",  # ty: ignore[invalid-argument-type]
        )

    # An explicit ``rng`` takes precedence over the backend's state.
    be = backends.Backend()
    run_sampler(be, seed=5)
    stored = be.random_state
    sampler = EnsembleSampler(
        be.shape[0], be.shape[1], normal_log_prob, backend=be, rng=42
    )
    assert (
        sampler.random_state == np.random.default_rng(42).bit_generator.state
    )
    assert stored is not None


def test_compute_log_prob_invalid_coords(nwalkers=32, ndim=3, seed=1234):
    coords = np.random.default_rng(seed).standard_normal((nwalkers, ndim))
    sampler = EnsembleSampler(nwalkers, ndim, normal_log_prob)

    coords_inf = np.array(coords)
    coords_inf[0, 0] = np.inf
    with pytest.raises(ValueError, match="infinite"):
        sampler.compute_log_prob(coords_inf)

    coords_nan = np.array(coords)
    coords_nan[0, 0] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        sampler.compute_log_prob(coords_nan)


def test_walkers_dependent_nonfinite(seed=1234):
    coords = np.random.default_rng(seed).standard_normal((10, 2))
    coords[0, 0] = np.inf
    assert not walkers_independent(coords)
    coords[0, 0] = np.nan
    assert not walkers_independent(coords)


@pytest.mark.parametrize("backend", all_backends)
def test_infinite_iterations_store(backend, nwalkers=32, ndim=3):
    with backend() as be:
        coords = np.random.default_rng(1234).standard_normal((nwalkers, ndim))
        with pytest.raises(ValueError):
            next(
                EnsembleSampler(
                    nwalkers, ndim, normal_log_prob, backend=be
                ).sample(coords, iterations=None, store=True)
            )


@pytest.mark.parametrize("backend", all_backends)
def test_infinite_iterations(backend, nwalkers=32, ndim=3):
    with backend() as be:
        coords = np.random.default_rng(1234).standard_normal((nwalkers, ndim))
        for _ in islice(
            EnsembleSampler(
                nwalkers, ndim, normal_log_prob, backend=be
            ).sample(coords, iterations=None, store=False),
            10,
        ):
            pass
