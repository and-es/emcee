import os
from itertools import product

import numpy as np
import pytest

from emcee import EnsembleSampler, State, backends
from emcee.backends.hdf import does_hdf5_support_longdouble

try:
    import h5py
except ImportError:
    h5py = None  # ty: ignore[invalid-assignment]

__all__ = ["test_backend", "test_reload"]

all_backends = backends.get_test_backends()
other_backends = all_backends[1:]
dtypes = [None, [("log_prior", float), ("mean", int)]]


def normal_log_prob(params):
    return -0.5 * np.sum(params**2)


def normal_log_prob_blobs(params):
    return normal_log_prob(params), 0.1, 5


def run_sampler(
    backend,
    nwalkers=32,
    ndim=3,
    nsteps=25,
    seed=1234,
    thin_by=1,
    dtype=None,
    blobs=True,
    lp=None,
    resume=False,
):
    if lp is None:
        lp = normal_log_prob_blobs if blobs else normal_log_prob
    coords = np.random.default_rng(seed).standard_normal((nwalkers, ndim))
    # When resuming, let the sampler restore its generator from the backend
    sampler = EnsembleSampler(
        nwalkers,
        ndim,
        lp,
        backend=backend,
        blobs_dtype=dtype,
        rng=None if resume else seed,
    )
    sampler.run_mcmc(coords, nsteps, thin_by=thin_by)
    return sampler


def _custom_allclose(a, b):
    if a.dtype.fields is None:
        assert np.allclose(a, b)
    else:
        for n in a.dtype.names:
            assert np.allclose(a[n], b[n])


@pytest.mark.skipif(h5py is None, reason="HDF5 not available")
def test_uninit(tmpdir):
    fn = str(tmpdir.join("EMCEE_TEST_FILE_DO_NOT_USE.h5"))
    if os.path.exists(fn):
        os.remove(fn)

    with backends.HDFBackend(fn) as be:
        run_sampler(be)

    assert os.path.exists(fn)
    os.remove(fn)


@pytest.mark.parametrize("backend", all_backends)
def test_uninit_errors(backend):
    with backend() as be:
        with pytest.raises(AttributeError):
            be.get_last_sample()

        for k in ["chain", "log_prob", "blobs"]:
            with pytest.raises(AttributeError):
                getattr(be, "get_" + k)()


@pytest.mark.parametrize("backend", all_backends)
def test_blob_usage_errors(backend):
    with backend() as be:
        run_sampler(be, blobs=True)
        with pytest.raises(ValueError):
            run_sampler(be, blobs=False)

    with backend() as be:
        run_sampler(be, blobs=False)
        with pytest.raises(ValueError):
            run_sampler(be, blobs=True)


@pytest.mark.parametrize(
    "backend,dtype,blobs",
    list(product(other_backends, dtypes, [True, False])),
)
def test_backend(backend, dtype, blobs):
    # Run a sampler with the default backend.
    sampler1 = run_sampler(backends.Backend(), dtype=dtype, blobs=blobs)

    with backend() as be:
        sampler2 = run_sampler(be, dtype=dtype, blobs=blobs)

        values = ["chain", "log_prob"]
        if blobs:
            values += ["blobs"]
        else:
            assert sampler1.get_blobs() is None
            assert sampler2.get_blobs() is None

        # Check all of the components.
        for k in values:
            a = getattr(sampler1, "get_" + k)()
            b = getattr(sampler2, "get_" + k)()
            _custom_allclose(a, b)

        last1 = sampler1.get_last_sample()
        last2 = sampler2.get_last_sample()
        assert np.allclose(last1.coords, last2.coords)
        assert np.allclose(last1.log_prob, last2.log_prob)
        assert last1.random_state == last2.random_state
        if blobs:
            _custom_allclose(last1.blobs, last2.blobs)
        else:
            assert last1.blobs is None and last2.blobs is None

        a = sampler1.acceptance_fraction
        b = sampler2.acceptance_fraction
        assert np.allclose(a, b), "inconsistent acceptance fraction"


@pytest.mark.parametrize(
    "backend,dtype", list(product(other_backends, dtypes))
)
def test_reload(backend, dtype):
    with backend() as backend1:
        run_sampler(backend1, dtype=dtype)

        # Test the state
        state = backend1.random_state
        assert isinstance(state, dict)

        # Load the file using a new backend object.
        backend2 = backends.HDFBackend(
            backend1.filename, backend1.name, read_only=True
        )

        with pytest.raises(RuntimeError):
            backend2.reset(32, 3)

        assert state == backend2.random_state

        # Check all of the components.
        for k in ["chain", "log_prob", "blobs"]:
            a = backend1.get_value(k)
            b = backend2.get_value(k)
            _custom_allclose(a, b)

        last1 = backend1.get_last_sample()
        last2 = backend2.get_last_sample()
        assert np.allclose(last1.coords, last2.coords)
        assert np.allclose(last1.log_prob, last2.log_prob)
        assert last1.random_state == last2.random_state
        _custom_allclose(last1.blobs, last2.blobs)

        a = backend1.accepted
        b = backend2.accepted
        assert np.allclose(a, b), "inconsistent accepted"


@pytest.mark.parametrize(
    "backend,dtype", list(product(other_backends, dtypes))
)
def test_restart(backend, dtype):
    # Run a sampler with the default backend.
    b = backends.Backend()
    run_sampler(b, dtype=dtype)
    sampler1 = run_sampler(b, seed=4321, dtype=dtype, resume=True)

    with backend() as be:
        run_sampler(be, dtype=dtype)
        sampler2 = run_sampler(be, seed=4321, dtype=dtype, resume=True)

        # Check all of the components.
        for k in ["chain", "log_prob", "blobs"]:
            a = getattr(sampler1, "get_" + k)()
            b = getattr(sampler2, "get_" + k)()
            _custom_allclose(a, b)

        last1 = sampler1.get_last_sample()
        last2 = sampler2.get_last_sample()
        assert np.allclose(last1.coords, last2.coords)
        assert np.allclose(last1.log_prob, last2.log_prob)
        assert last1.random_state == last2.random_state
        _custom_allclose(last1.blobs, last2.blobs)

        a = sampler1.acceptance_fraction
        b = sampler2.acceptance_fraction
        assert np.allclose(a, b), "inconsistent acceptance fraction"


@pytest.mark.skipif(h5py is None, reason="HDF5 not available")
def test_resume_random_state():
    # The generator state stored by the backend is restored on resume
    with backends.TempHDFBackend() as b:
        run_sampler(b, nsteps=5)
        state = b.random_state
        assert isinstance(state, dict)
        assert state["bit_generator"] == "PCG64"

        sampler = EnsembleSampler(32, 3, normal_log_prob_blobs, backend=b)
        assert sampler.random_state == state


@pytest.mark.skipif(h5py is None, reason="HDF5 not available")
def test_legacy_random_state_migration():
    # A file written by an older emcee stores the random state as
    # ``random_state_{i}`` attributes; it must be readable and migrated
    # to the new format on the next save
    with backends.TempHDFBackend() as b:
        run_sampler(b, nsteps=3)
        legacy = np.random.mtrand.RandomState(11).get_state()
        with h5py.File(b.filename, "a") as f:
            g = f[b.name]
            del g.attrs["random_state"]
            for i, v in enumerate(legacy):
                g.attrs[f"random_state_{i}"] = v

        # The legacy attributes are returned as a list
        stored = b.random_state
        assert stored[0] == "MT19937"

        # Resuming converts the legacy state to an MT19937 generator
        sampler = EnsembleSampler(32, 3, normal_log_prob_blobs, backend=b)
        state = sampler.random_state
        assert state["bit_generator"] == "MT19937"
        # numpy stubs type get_state() as a dict, but legacy=True
        # returns a tuple
        assert np.array_equal(
            state["state"]["key"],
            legacy[1],  # ty: ignore[invalid-argument-type]
        )
        assert state["state"]["pos"] == legacy[2]  # ty: ignore[invalid-argument-type]

        # After another step the file uses the new format only
        sampler.run_mcmc(None, 1)
        with h5py.File(b.filename, "r") as f:
            attrs = set(f[b.name].attrs)
            assert "random_state" in attrs
            assert not any(a.startswith("random_state_") for a in attrs)


@pytest.mark.skipif(h5py is None, reason="HDF5 not available")
@pytest.mark.filterwarnings("error::RuntimeWarning")
def test_corrupt_random_state_warns_on_resume():
    # A stored state that cannot be restored (truncated legacy attrs or
    # an unknown bit generator) must warn exactly once and fall back to
    # a fresh generator instead of crashing construction
    import json

    # Truncated legacy attribute set
    with backends.TempHDFBackend() as b:
        run_sampler(b, nsteps=3)
        with h5py.File(b.filename, "a") as f:
            g = f[b.name]
            del g.attrs["random_state"]
            g.attrs["random_state_0"] = "MT19937"
        with pytest.warns(RuntimeWarning, match="could not be restored"):
            sampler = EnsembleSampler(32, 3, normal_log_prob_blobs, backend=b)
        sampler.run_mcmc(None, 1)

    # Unknown bit generator name in the new format
    with backends.TempHDFBackend() as b:
        run_sampler(b, nsteps=3)
        with h5py.File(b.filename, "a") as f:
            f[b.name].attrs["random_state"] = json.dumps(
                {"bit_generator": "NotABitGenerator", "state": {}}
            )
        with pytest.warns(RuntimeWarning, match="could not be restored"):
            sampler = EnsembleSampler(32, 3, normal_log_prob_blobs, backend=b)
        sampler.run_mcmc(None, 1)


def test_json_default_numpy_scalars():
    # Bit generator states from third-party generators may contain numpy
    # scalar types beyond int/ndarray
    import json

    from emcee.backends.hdf import _json_default

    payload = {
        "i": np.int64(3),
        "f": np.float64(1.5),
        "b": np.bool_(True),
        "arr": np.arange(3, dtype=np.uint32),
    }
    result = json.loads(json.dumps(payload, default=_json_default))
    assert result == {"i": 3, "f": 1.5, "b": True, "arr": [0, 1, 2]}

    with pytest.raises(TypeError, match="not JSON serializable"):
        json.dumps({"x": object()}, default=_json_default)


@pytest.mark.skipif(h5py is None, reason="HDF5 not available")
def test_multi_hdf5():
    with backends.TempHDFBackend() as backend1:
        run_sampler(backend1)

        backend2 = backends.HDFBackend(backend1.filename, name="mcmc2")
        run_sampler(backend2)
        chain2 = backend2.get_chain()

        with h5py.File(backend1.filename, "r") as f:
            assert set(f.keys()) == {backend1.name, "mcmc2"}

        backend1.reset(10, 2)
        assert np.allclose(backend2.get_chain(), chain2)
        with pytest.raises(AttributeError):
            backend1.get_chain()


@pytest.mark.parametrize("backend", all_backends)
def test_longdouble_preserved(backend):
    if (
        issubclass(backend, backends.TempHDFBackend)
        and not does_hdf5_support_longdouble()
    ):
        pytest.xfail("HDF5 does not support long double on this platform")
    nwalkers = 10
    ndim = 2
    nsteps = 5
    with backend(dtype=np.longdouble) as b:
        b.reset(nwalkers, ndim)
        b.grow(nsteps, None)
        for i in range(nsteps):
            coords = np.zeros((nwalkers, ndim), dtype=np.longdouble)
            coords += i + 1
            coords += np.arange(nwalkers)[:, None]
            coords[:, 1] += coords[:, 0] * 2 * np.finfo(np.longdouble).eps
            assert not np.any(coords[:, 1] == coords[:, 0])
            lp = 1 + np.arange(nwalkers) * np.finfo(np.longdouble).eps
            state = State(coords, log_prob=lp, random_state=())
            b.save_step(state, np.ones((nwalkers,), dtype=bool))
            s = b.get_last_sample()
            # check s has adequate precision and equals state
            assert s.coords.dtype == np.longdouble
            assert not np.any(s.coords[:, 1] == s.coords[:, 0])
            assert np.all(s.coords == coords)

            assert s.log_prob.dtype == np.longdouble
            assert np.all(s.log_prob == lp)


@pytest.mark.parametrize("backend", all_backends)
def test_save_step_validation(backend):
    nwalkers, ndim, nsteps = 8, 2, 3
    coords = np.zeros((nwalkers, ndim))
    log_prob = np.zeros(nwalkers)
    blobs = np.zeros(nwalkers)
    accepted = np.ones(nwalkers, dtype=bool)

    def make_state(c=None, lp=None, b=blobs):
        return State(
            coords if c is None else c,
            log_prob=log_prob if lp is None else lp,
            blobs=b,
            random_state=(),
        )

    with backend() as be:
        be.reset(nwalkers, ndim)
        be.grow(nsteps, blobs)

        with pytest.raises(ValueError, match="invalid coordinate dimensions"):
            be.save_step(
                make_state(c=np.zeros((nwalkers, ndim + 1))), accepted
            )
        with pytest.raises(ValueError, match="invalid log probability size"):
            be.save_step(make_state(lp=np.zeros(nwalkers - 1)), accepted)
        with pytest.raises(ValueError, match="invalid blobs size"):
            be.save_step(make_state(b=np.zeros(nwalkers - 1)), accepted)
        with pytest.raises(ValueError, match="inconsistent use of blobs"):
            be.save_step(make_state(b=None), accepted)
        with pytest.raises(ValueError, match="invalid acceptance size"):
            be.save_step(make_state(), np.ones(nwalkers - 1, dtype=bool))

        # A well-formed state must still be accepted.
        be.save_step(make_state(), accepted)
        assert be.iteration == 1

    # A backend grown without blobs must reject states carrying blobs.
    with backend() as be:
        be.reset(nwalkers, ndim)
        be.grow(nsteps, None)
        with pytest.raises(ValueError, match="unexpected blobs"):
            be.save_step(make_state(), accepted)


@pytest.mark.skipif(h5py is None, reason="HDF5 not available")
def test_hdf5_compression():
    with backends.TempHDFBackend(compression="gzip") as b:
        run_sampler(b, blobs=True)
        # re-open and read
        b.get_chain()
        b.get_blobs()
        b.get_log_prob()
        assert b.accepted is not None
