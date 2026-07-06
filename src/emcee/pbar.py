from __future__ import annotations

import importlib
import logging
from typing import Any, Protocol

__all__ = ["get_progress_bar"]

logger = logging.getLogger(__name__)

try:
    import tqdm
    import tqdm.auto
except ImportError:
    tqdm = None  # ty: ignore[invalid-assignment]


class ProgressBar(Protocol):
    """The interface of the progress bars used by the sampler"""

    def __enter__(self) -> ProgressBar: ...

    def __exit__(
        self, exc_type: object, exc_value: object, traceback: object, /
    ) -> object: ...

    def update(self, n: int, /) -> object: ...


class _NoOpPBar:
    """This class implements the progress bar interface but does nothing"""

    def __enter__(self) -> _NoOpPBar:
        return self

    def __exit__(self, *args: object) -> None:
        pass

    def update(self, count: int) -> None:
        pass


def get_progress_bar(
    display: bool | str, total: int | None, **kwargs: Any
) -> ProgressBar:
    """Get a progress bar interface with given properties

    If the tqdm library is not installed, this will always return a "progress
    bar" that does nothing.

    Args:
        display (bool or str): Should the bar actually show the progress? Or
                               a string to indicate which tqdm bar
                               (submodule) to use.
        total (int): The total size of the progress bar.
        kwargs (dict): Optional keyword arguments to be passed to the tqdm
                       call.

    """
    if display:
        if tqdm is None:
            logger.warning(
                "You must install the tqdm library to use progress "
                "indicators with emcee"
            )
            return _NoOpPBar()
        else:
            if display is True:
                return tqdm.auto.tqdm(total=total, **kwargs)
            else:
                tqdm_submodule = importlib.import_module(f"tqdm.{display}")
                return tqdm_submodule.tqdm(total=total, **kwargs)

    return _NoOpPBar()
