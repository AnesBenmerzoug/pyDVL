import os
from abc import ABCMeta, abstractmethod
from dataclasses import asdict
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Tuple,
    Type,
    TypeVar,
    Union,
)

import joblib
import ray
from joblib import delayed
from ray import ObjectRef
from ray.util.joblib import register_ray

from ..config import ParallelConfig

__all__ = ["init_parallel_backend", "effective_n_jobs", "available_cpus"]

T = TypeVar("T")

_PARALLEL_BACKENDS: Dict[str, "Type[BaseParallelBackend]"] = {}


class NoPublicConstructor(ABCMeta):
    """Metaclass that ensures a private constructor

    If a class uses this metaclass like this:

        class SomeClass(metaclass=NoPublicConstructor):
            pass

    If you try to instantiate your class (`SomeClass()`),
    a `TypeError` will be thrown.

    Taken almost verbatim from:
    https://stackoverflow.com/a/64682734
    """

    def __call__(cls, *args, **kwargs):
        raise TypeError(
            f"{cls.__module__}.{cls.__qualname__} cannot be initialized directly. "
            "Use init_parallel_backend() instead."
        )

    def _create(cls, *args: Any, **kwargs: Any):
        return super().__call__(*args, **kwargs)


class BaseParallelBackend(metaclass=NoPublicConstructor):
    """Abstract base class for all parallel backends"""

    config: Dict[str, Any] = {}

    def __init_subclass__(cls, *, backend_name: str, **kwargs):
        global _PARALLEL_BACKENDS
        _PARALLEL_BACKENDS[backend_name] = cls
        super().__init_subclass__(**kwargs)

    @abstractmethod
    def get(self, v: Any, *args, **kwargs):
        ...

    @abstractmethod
    def put(self, v: Any, *args, **kwargs) -> Any:
        ...

    @abstractmethod
    def wrap(self, fun: Callable, **kwargs) -> Callable:
        ...

    @abstractmethod
    def wait(self, v: Any, *args, **kwargs) -> Any:
        ...

    @abstractmethod
    def _effective_n_jobs(self, n_jobs: int) -> int:
        ...

    def effective_n_jobs(self, n_jobs: int = -1) -> int:
        if n_jobs == 0:
            raise ValueError("n_jobs == 0 in Parallel has no meaning")
        n_jobs = self._effective_n_jobs(n_jobs)
        return n_jobs

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}: {self.config}>"


class JoblibParallelBackend(BaseParallelBackend, backend_name="joblib"):
    """Class used to wrap joblib to make it transparent to algorithms.

    It shouldn't be initialized directly. You should instead call
    :func:`~pydvl.utils.parallel.backend.init_parallel_backend`.

    :param config: instance of :class:`~pydvl.utils.config.ParallelConfig` with
        cluster address, number of cpus, etc.
    """

    def __init__(self, config: ParallelConfig):
        self.config = {
            "logging_level": config.logging_level,
            "n_jobs": config.n_cpus_local,
        }

    def get(
        self,
        v: T,
        *args,
        **kwargs,
    ) -> T:
        return v

    def put(self, v: T, *args, **kwargs) -> T:
        return v

    def wrap(self, fun: Callable, **kwargs) -> Callable:
        """Wraps a function as a joblib delayed.

        :param fun: the function to wrap

        :return: The delayed function.
        """
        return delayed(fun)  # type: ignore

    def wait(
        self,
        v: List[T],
        *args,
        **kwargs,
    ) -> Tuple[List[T], List[T]]:
        return v, []

    def _effective_n_jobs(self, n_jobs: int) -> int:
        if self.config["n_jobs"] is None:
            maximum_n_jobs = joblib.effective_n_jobs()
        else:
            maximum_n_jobs = self.config["n_jobs"]
        eff_n_jobs: int = min(joblib.effective_n_jobs(n_jobs), maximum_n_jobs)
        return eff_n_jobs


class RayParallelBackend(BaseParallelBackend, backend_name="ray"):
    """Class used to wrap ray to make it transparent to algorithms.

    It shouldn't be initialized directly. You should instead call
    :func:`~pydvl.utils.parallel.backend.init_parallel_backend`.

    :param config: instance of :class:`~pydvl.utils.config.ParallelConfig` with
        cluster address, number of cpus, etc.
    """

    def __init__(self, config: ParallelConfig):
        self.config = {"address": config.address, "logging_level": config.logging_level}
        if self.config["address"] is None:
            self.config["num_cpus"] = config.n_cpus_local
        if not ray.is_initialized():
            ray.init(**self.config)
        # Register ray joblib backend
        register_ray()

    def get(
        self,
        v: Union[ObjectRef, Iterable[ObjectRef], T],
        *args,
        **kwargs,
    ) -> Union[T, Any]:
        timeout: Optional[float] = kwargs.get("timeout", None)
        if isinstance(v, ObjectRef):
            return ray.get(v, timeout=timeout)
        elif isinstance(v, Iterable):
            return [self.get(x, timeout=timeout) for x in v]
        else:
            return v

    def put(self, v: T, *args, **kwargs) -> Union["ObjectRef[T]", T]:
        try:
            return ray.put(v, **kwargs)  # type: ignore
        except TypeError:
            return v  # type: ignore

    def wrap(self, fun: Callable, **kwargs) -> Callable:
        """Wraps a function as a ray remote.

        :param fun: the function to wrap
        :param kwargs: keyword arguments to pass to @ray.remote

        :return: The `.remote` method of the ray `RemoteFunction`.
        """
        if len(kwargs) > 0:
            return ray.remote(**kwargs)(fun).remote  # type: ignore
        return ray.remote(fun).remote  # type: ignore

    def wait(
        self,
        v: List["ObjectRef"],
        *args,
        **kwargs,
    ) -> Tuple[List[ObjectRef], List[ObjectRef]]:
        num_returns: int = kwargs.get("num_returns", 1)
        timeout: Optional[float] = kwargs.get("timeout", None)
        return ray.wait(  # type: ignore
            v,
            num_returns=num_returns,
            timeout=timeout,
        )

    def _effective_n_jobs(self, n_jobs: int) -> int:
        ray_cpus = int(ray._private.state.cluster_resources()["CPU"])  # type: ignore
        if n_jobs < 0:
            eff_n_jobs = ray_cpus
        else:
            eff_n_jobs = min(n_jobs, ray_cpus)
        return eff_n_jobs


def init_parallel_backend(
    config: ParallelConfig,
) -> BaseParallelBackend:
    """Initializes the parallel backend and returns an instance of it.

    :param config: instance of :class:`~pydvl.utils.config.ParallelConfig`
        with cluster address, number of cpus, etc.

    :Example:

    >>> from pydvl.utils.parallel.backend import init_parallel_backend
    >>> from pydvl.utils.config import ParallelConfig
    >>> config = ParallelConfig()
    >>> parallel_backend = init_parallel_backend(config)
    >>> parallel_backend
    <JoblibParallelBackend: {'logging_level': 30, 'n_jobs': None}>

    >>> from pydvl.utils.parallel.backend import init_parallel_backend
    >>> from pydvl.utils.config import ParallelConfig
    >>> config = ParallelConfig(backend="ray")
    >>> parallel_backend = init_parallel_backend(config)
    >>> parallel_backend
    <RayParallelBackend: {'address': None, 'logging_level': 30, 'num_cpus': None}>

    """
    try:
        parallel_backend_cls = _PARALLEL_BACKENDS[config.backend]
    except KeyError:
        raise NotImplementedError(f"Unexpected parallel backend {config.backend}")
    parallel_backend = parallel_backend_cls._create(config)
    return parallel_backend  # type: ignore


def available_cpus() -> int:
    """Platform-independent count of available cores.

    FIXME: do we really need this or is `os.cpu_count` enough? Is this portable?
    :return: Number of cores, or 1 if it is not possible to determine.
    """
    from platform import system

    if system() != "Linux":
        return os.cpu_count() or 1
    return len(os.sched_getaffinity(0))  # type: ignore


def effective_n_jobs(n_jobs: int, config: ParallelConfig = ParallelConfig()) -> int:
    """Returns the effective number of jobs.

    This number may vary depending on the parallel backend and the resources
    available.

    :param n_jobs: the number of jobs requested. If -1, the number of available
        CPUs is returned.
    :param config: instance of :class:`~pydvl.utils.config.ParallelConfig` with
        cluster address, number of cpus, etc.
    :return: the effective number of jobs, guaranteed to be >= 1.
    :raises RuntimeError: if the effective number of jobs returned by the backend
        is < 1.
    """
    parallel_backend = init_parallel_backend(config)
    if (eff_n_jobs := parallel_backend.effective_n_jobs(n_jobs)) < 1:
        raise RuntimeError(
            f"Invalid number of jobs {eff_n_jobs} obtained from parallel backend {config.backend}"
        )
    return eff_n_jobs
