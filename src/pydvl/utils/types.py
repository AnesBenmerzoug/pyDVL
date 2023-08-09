""" This module contains types, protocols, decorators and generic function
transformations. Some of it probably belongs elsewhere.
"""
import inspect
import numbers
from typing import Any, Callable, Optional, Protocol, TypeVar, Union

import numpy as np
from numpy.random import SeedSequence
from numpy.typing import NDArray

__all__ = [
    "SupervisedModel",
    "MapFunction",
    "ReduceFunction",
    "Seed",
    "SeedOrGenerator",
    "check_seed",
]

R = TypeVar("R", covariant=True)


class MapFunction(Protocol[R]):
    def __call__(self, *args: Any, **kwargs: Any) -> R:
        ...


class ReduceFunction(Protocol[R]):
    def __call__(self, *args: Any, **kwargs: Any) -> R:
        ...


class SupervisedModel(Protocol):
    """This is the minimal Protocol that valuation methods require from
    models in order to work.

    All that is needed are the standard sklearn methods `fit()`, `predict()` and
    `score()`.
    """

    def fit(self, x: NDArray, y: NDArray):
        pass

    def predict(self, x: NDArray) -> NDArray:
        pass

    def score(self, x: NDArray, y: NDArray) -> float:
        pass


def maybe_add_argument(fun: Callable, new_arg: str):
    """Wraps a function to accept the given keyword parameter if it doesn't
    already.

    If `fun` already takes a keyword parameter of name `new_arg`, then it is
    returned as is. Otherwise, a wrapper is returned which merely ignores the
    argument.

    :param fun: The function to wrap
    :param new_arg: The name of the argument that the new function will accept
        (and ignore).
    :return: A new function accepting one more keyword argument.
    """
    params = inspect.signature(fun).parameters
    if new_arg in params.keys():
        return fun

    def wrapper(*args, **kwargs):
        try:
            del kwargs[new_arg]
        except KeyError:
            pass
        return fun(*args, **kwargs)

    return wrapper


Seed = Optional[Union[int, np.random.SeedSequence]]
SeedOrGenerator = Union[Seed, np.random.Generator]


def check_seed(seed: Seed, return_none: bool = True) -> Optional[SeedSequence]:
    """Check if the seed is valid and return a SeedSequence object if it is. If it is
    not valid, return None."""

    if seed is None:
        if return_none:
            return None
        else:
            return SeedSequence()

    elif isinstance(seed, int):
        return SeedSequence(seed)
    else:
        return seed
