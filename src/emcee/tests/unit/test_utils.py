import pytest

from emcee.utils import deprecated, deprecation_warning

__all__ = [
    "test_deprecation_warning",
    "test_deprecated_decorator",
]


def test_deprecation_warning():
    with pytest.warns(DeprecationWarning, match="something is deprecated"):
        deprecation_warning("something is deprecated")


def test_deprecated_decorator():
    @deprecated("new_func")
    def old_func(x):
        return 2 * x

    with pytest.warns(
        DeprecationWarning, match=r"'old_func' is deprecated. Use 'new_func'"
    ):
        assert old_func(21) == 42


def test_deprecated_decorator_no_alternate():
    @deprecated(None)
    def old_func():
        return "value"

    with pytest.warns(DeprecationWarning, match=r"'old_func' is deprecated."):
        assert old_func() == "value"
