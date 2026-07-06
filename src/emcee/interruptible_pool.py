# The standard library now has an interruptible pool
from multiprocessing.pool import Pool as InterruptiblePool

from .utils import deprecation_warning

__all__ = ["InterruptiblePool"]

deprecation_warning(
    "The 'emcee.interruptible_pool' module is deprecated; use "
    "'multiprocessing.pool.Pool' from the standard library instead"
)
