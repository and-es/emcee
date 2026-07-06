from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import ArrayLike

__all__ = ["function_1d", "integrated_time", "AutocorrError"]

logger = logging.getLogger(__name__)


def next_pow_two(n: int) -> int:
    """Returns the next power of two greater than or equal to `n`"""
    i = 1
    while i < n:
        i = i << 1
    return i


def function_1d(x: ArrayLike) -> np.ndarray:
    """Estimate the normalized autocorrelation function of a 1-D series

    Args:
        x: The series as a 1-D numpy array.

    Returns:
        array: The autocorrelation function of the time series.

    """
    x = np.atleast_1d(x)
    if len(x.shape) != 1:
        raise ValueError("invalid dimensions for 1D autocorrelation function")
    n = next_pow_two(len(x))

    # Compute the FFT and then (from that) the auto-correlation function
    f = np.fft.fft(x - np.mean(x), n=2 * n)
    acf = np.fft.ifft(f * np.conjugate(f))[: len(x)].real
    acf /= acf[0]
    return acf


def auto_window(taus: np.ndarray, c: float) -> int:
    m = np.arange(len(taus)) < c * taus
    if np.any(m):
        return int(np.argmin(m))
    return len(taus) - 1


def integrated_time(
    x: ArrayLike,
    c: float = 5,
    tol: float = 50,
    quiet: bool = False,
    has_walkers: bool = True,
) -> np.ndarray:
    """Estimate the integrated autocorrelation time of a time series.

    This estimate uses the iterative procedure described on page 16 of
    `Sokal's notes <https://www.semanticscholar.org/paper/Monte-Carlo-Methods-in-Statistical-Mechanics%3A-and-Sokal/0bfe9e3db30605fe2d4d26e1a288a5e2997e7225>`_ to
    determine a reasonable window size.

    Args:
        x (numpy.ndarray): The time series. If 2-dimensional, the array
            dimesions are interpreted as ``(n_step, n_walker)`` unless
            ``has_walkers==False``, in which case they are interpreted as
            ``(n_step, n_param)``. If 3-dimensional, the dimensions are
            interperted as ``(n_step, n_walker, n_param)``.
        c (Optional[float]): The step size for the window search. (default:
            ``5``)
        tol (Optional[float]): The minimum number of autocorrelation times
            needed to trust the estimate. (default: ``50``)
        quiet (Optional[bool]): This argument controls the behavior when the
            chain is too short. If ``True``, give a warning instead of raising
            an :class:`AutocorrError`. (default: ``False``)
        has_walkers (Optional[bool]): Whether the last axis should be
            interpreted as walkers or parameters if ``x`` has 2 dimensions.
            (default: ``True``)

    Returns:
        float or array: An estimate of the integrated autocorrelation time of
            the time series ``x``.

    Raises
        AutocorrError: If the autocorrelation time can't be reliably estimated
            from the chain and ``quiet`` is ``False``. This normally means
            that the chain is too short.

    """
    x = np.atleast_1d(x)
    if len(x.shape) == 1:
        x = x[:, np.newaxis, np.newaxis]
    if len(x.shape) == 2:
        if not has_walkers:
            x = x[:, np.newaxis, :]
        else:
            x = x[:, :, np.newaxis]
    if len(x.shape) != 3:
        raise ValueError("invalid dimensions")

    n_t, n_w, n_d = x.shape

    # Compute the autocorrelation function for all walkers and parameters
    # at once with a batched real FFT. The time axis is moved last and the
    # array copied into C order so that the transforms run along contiguous
    # memory; the copy also keeps the caller's array untouched.
    y = np.moveaxis(x, 0, -1).astype(np.float64, order="C")
    y -= np.mean(y, axis=-1, keepdims=True)
    n = next_pow_two(n_t)
    f = np.fft.rfft(y, n=2 * n, axis=-1)
    acf = np.fft.irfft(f * np.conjugate(f), n=2 * n, axis=-1)[..., :n_t]
    acf /= acf[..., :1]
    taus = 2.0 * np.cumsum(np.mean(acf, axis=0), axis=-1) - 1.0

    tau_est = np.empty(n_d)
    windows = np.empty(n_d, dtype=int)
    for d in range(n_d):
        windows[d] = auto_window(taus[d], c)
        tau_est[d] = taus[d, windows[d]]

    # Check convergence
    flag = tol * tau_est > n_t

    # Warn or raise in the case of non-convergence
    if np.any(flag):
        msg = (
            f"The chain is shorter than {tol} times the integrated "
            f"autocorrelation time for {np.sum(flag)} parameter(s). Use "
            "this estimate with caution and run a longer chain!\n"
        )
        msg += f"N/{tol} = {n_t / tol:.0f};\ntau: {tau_est}"
        if not quiet:
            raise AutocorrError(tau_est, msg)
        logger.warning(msg)

    return tau_est


class AutocorrError(Exception):
    """Raised if the chain is too short to estimate an autocorrelation time.

    The current estimate of the autocorrelation time can be accessed via the
    ``tau`` attribute of this exception.

    """

    tau: np.ndarray

    def __init__(self, tau: np.ndarray, *args: Any, **kwargs: Any) -> None:
        self.tau = tau
        super().__init__(*args, **kwargs)
