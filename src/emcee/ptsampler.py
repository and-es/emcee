from .utils import deprecation_warning

try:
    from ptemcee import Sampler as PTSampler
except ImportError:

    class PTSampler:
        def __init__(self, *args, **kwargs):
            raise ImportError(
                "The PTSampler from emcee has been forked to "
                "https://github.com/willvousden/ptemcee, "
                "please install that package to continue using the PTSampler"
            )


__all__ = ["PTSampler"]

deprecation_warning(
    "The 'emcee.ptsampler' module is deprecated and will be removed in "
    "the next major version; use the 'ptemcee' package "
    "(https://github.com/willvousden/ptemcee) instead"
)
