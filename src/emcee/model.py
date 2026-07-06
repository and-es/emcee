from __future__ import annotations

from typing import TYPE_CHECKING, Any, NamedTuple

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

__all__ = ["Model"]


class Model(NamedTuple):
    """The model functions and random number generator used by the moves"""

    log_prob_fn: Callable[[Any], Any] | None
    compute_log_prob_fn: Callable[
        [np.ndarray], tuple[np.ndarray, np.ndarray | None]
    ]
    map_fn: Callable[..., Iterable[Any]]
    random: np.random.Generator
