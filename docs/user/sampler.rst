.. _sampler:

The Ensemble Sampler
====================

Standard usage of ``emcee`` involves instantiating an
:class:`EnsembleSampler`.

For reproducible results, pass a seed or a ``numpy.random.Generator`` as
the ``rng`` argument:

.. code-block:: python

    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prob, rng=42)

Note that the global ``numpy.random.seed`` has no effect on the sampler;
the sampler owns its own ``numpy.random.Generator``, which is also
persisted by the backends so that interrupted runs can be resumed
deterministically.

.. autoclass:: emcee.EnsembleSampler
   :inherited-members:

Note that several of the :class:`EnsembleSampler` methods return or consume
:class:`State` objects:

.. autoclass:: emcee.State
   :inherited-members:
