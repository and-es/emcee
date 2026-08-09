from emcee import moves

from .test_proposal import _test_normal, _test_uniform

__all__ = ["test_normal_walk", "test_normal_walk_no_s", "test_uniform_walk"]


def test_normal_walk(**kwargs):
    _test_normal(moves.WalkMove(s=3), **kwargs)


def test_normal_walk_no_s(**kwargs):
    # The default s=None uses the full complement, which is a separate
    # code path from a fixed-size subset.
    _test_normal(moves.WalkMove(), **kwargs)


def test_uniform_walk(**kwargs):
    _test_uniform(moves.WalkMove(s=3), **kwargs)
