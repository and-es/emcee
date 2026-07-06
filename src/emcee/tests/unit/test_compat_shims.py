import importlib
import sys

import pytest

__all__ = ["test_shim_import_warns"]


@pytest.mark.parametrize(
    "module",
    ["emcee.interruptible_pool", "emcee.mpi_pool", "emcee.ptsampler"],
)
def test_shim_import_warns(module):
    # The warning is emitted at import time, so force a fresh import
    sys.modules.pop(module, None)
    try:
        with pytest.warns(DeprecationWarning, match="deprecated"):
            importlib.import_module(module)
    finally:
        sys.modules.pop(module, None)
