import numpy as np
import pytest

from emcee import EnsembleSampler, backends

__all__ = ["test_blob_shape"]

# A generator for the random blob values returned by the test models
blob_rng = np.random.default_rng(42)


class BlobLogProb:
    def __init__(self, blob_function):
        self.blob_function = blob_function

    def __call__(self, params):
        return -0.5 * np.sum(params**2), self.blob_function(params)


@pytest.mark.parametrize("backend", backends.get_test_backends())
@pytest.mark.parametrize(
    "blob_spec",
    [
        (True, False, 5, lambda x: blob_rng.standard_normal(5)),
        (True, False, (5, 3), lambda x: blob_rng.standard_normal((5, 3))),
        (
            True,
            False,
            (5, 3),
            lambda x: blob_rng.standard_normal((1, 5, 1, 3, 1)),
        ),
        (True, False, 0, lambda x: blob_rng.standard_normal()),
        (False, True, 2, lambda x: (1.0, blob_rng.standard_normal(3))),
        (False, False, 0, lambda x: "face"),
        (False, False, 0, lambda x: object()),
        (False, False, 2, lambda x: ("face", "surface")),
        (False, True, 2, lambda x: (blob_rng.standard_normal(5), "face")),
    ],
)
def test_blob_shape(backend, blob_spec):
    # HDF backends don't support the object type
    hdf_able, ragged, blob_shape, func = blob_spec
    if backend in (backends.TempHDFBackend,) and not hdf_able:
        return

    with backend() as be:
        model = BlobLogProb(func)
        coords = np.random.default_rng(42).standard_normal((32, 3))
        nwalkers, ndim = coords.shape

        sampler = EnsembleSampler(nwalkers, ndim, model, backend=be)
        nsteps = 10

        sampler.run_mcmc(coords, nsteps)

        shape = [nsteps, nwalkers]
        if isinstance(blob_shape, tuple):
            shape += blob_shape
        elif blob_shape > 0:
            shape += [blob_shape]

        blobs = sampler.get_blobs()
        assert blobs is not None
        assert blobs.shape == tuple(shape)
        if not hdf_able:
            assert blobs.dtype == np.dtype("object")


class VariableLogProb:
    def __init__(self):
        self.i = 3

    def __call__(self, *args):
        return 0, np.zeros(self.i)


@pytest.mark.parametrize("backend", backends.get_test_backends())
def test_blob_mismatch(backend):
    with backend() as be:
        model = VariableLogProb()
        coords = np.random.default_rng(42).standard_normal((32, 3))
        nwalkers, ndim = coords.shape

        sampler = EnsembleSampler(nwalkers, ndim, model, backend=be)

        model.i += 1
        # We don't save blobs from the initial points
        # so blob shapes are taken from the first round of moves
        sampler.run_mcmc(coords, 1)

        model.i += 1
        with pytest.raises(ValueError):
            sampler.run_mcmc(coords, 1)


def test_blob_inconsistent_presence():
    # Blobs returned for only some walkers must raise instead of silently
    # dropping entries and misaligning the blob array.
    def log_prob(x):
        if x[0] > 0:
            return 0.0, 1.0
        return (0.0,)

    coords = np.random.default_rng(42).standard_normal((32, 3))
    coords[0, 0] = -np.abs(coords[0, 0])
    coords[1, 0] = np.abs(coords[1, 0])

    sampler = EnsembleSampler(32, 3, log_prob)
    with pytest.raises(ValueError, match="some walkers but not others"):
        sampler.compute_log_prob(coords)
