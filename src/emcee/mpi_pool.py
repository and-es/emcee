from .utils import deprecation_warning

try:
    from schwimmbad import MPIPool
except ImportError:

    class MPIPool:
        def __init__(self, *args, **kwargs):
            raise ImportError(
                "The MPIPool from emcee has been forked to "
                "https://github.com/adrn/schwimmbad, "
                "please install that package to continue using the MPIPool"
            )


__all__ = ["MPIPool"]

deprecation_warning(
    "The 'emcee.mpi_pool' module is deprecated; use the 'schwimmbad' "
    "package (https://github.com/adrn/schwimmbad) instead"
)
