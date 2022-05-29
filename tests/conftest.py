from collections import OrderedDict
from typing import Type

import numpy as np
import pytest
from sklearn.linear_model import LinearRegression

from valuation.utils import Dataset, Utility
from valuation.utils.numeric import spearman


def is_memcache_responsive(hostname, port):
    from pymemcache.client import Client

    try:
        client = Client(server=(hostname, port))
        client.flush_all()
        return True
    except ConnectionRefusedError:
        return False


@pytest.fixture(scope="session")
def memcached_service(docker_ip, docker_services):
    """Ensure that memcached service is up and responsive."""
    # `port_for` takes a container port and returns the corresponding host port
    port = docker_services.port_for("memcached", 11211)
    hostname, port = docker_ip, port
    docker_services.wait_until_responsive(
        timeout=30.0, pause=0.5, check=lambda: is_memcache_responsive(hostname, port)
    )
    return hostname, port


@pytest.fixture(scope="session")
def memcache_client_config(memcached_service):
    from valuation.utils import ClientConfig

    client_config = ClientConfig(
        server=memcached_service, connect_timeout=1.0, timeout=0.1, no_delay=True
    )
    return client_config


@pytest.fixture(scope="function")
def memcached_client(memcache_client_config):
    from pymemcache.client import Client

    try:
        c = Client(**memcache_client_config)
        c.flush_all()
        return c, memcache_client_config
    except Exception as e:
        print(
            f"Could not connect to memcached server "
            f'{memcache_client_config["server"]}: {e}'
        )
        raise e


@pytest.fixture(scope="session")
def boston_dataset():
    from sklearn import datasets

    return Dataset.from_sklearn(datasets.load_boston())


@pytest.fixture(scope="session")
def linear_dataset():
    from sklearn.utils import Bunch

    a = 2
    b = 0
    x = np.arange(-1, 1, 0.15)
    y = np.random.normal(loc=a * x + b, scale=0.1)
    db = Bunch()
    db.data, db.target = x.reshape(-1, 1), y
    db.DESCR = f"y~N({a}*x + {b}, 1)"
    db.feature_names = ["x"]
    db.target_names = ["y"]
    return Dataset.from_sklearn(data=db, train_size=0.66)


def polynomial(coefficients, x):
    powers = np.arange(len(coefficients))
    return np.power(x, np.tile(powers, (len(x), 1)).T).T @ coefficients


@pytest.fixture(scope="function")
def polynomial_dataset(coefficients: np.ndarray):
    """Coefficients must be for monomials of increasing degree"""
    from sklearn.utils import Bunch

    x = np.arange(-1, 1, 0.1)
    locs = polynomial(coefficients, x)
    y = np.random.normal(loc=locs, scale=0.1)
    db = Bunch()
    db.data, db.target = x.reshape(-1, 1), y
    poly = [f"{c} x^{i}" for i, c in enumerate(coefficients)]
    poly = " + ".join(poly)
    db.DESCR = f"$y \\sim N({poly}, 1)$"
    db.feature_names = ["x"]
    db.target_names = ["y"]
    return Dataset.from_sklearn(data=db, train_size=0.5), coefficients


@pytest.fixture()
def scoring():
    return "r2"


def dummy_utility(num_samples: int = 10):
    from numpy import ndarray

    from valuation.utils import SupervisedModel

    # Indices match values
    x = np.arange(0, num_samples, 1).reshape(-1, 1)
    nil = np.zeros_like(x)
    data = Dataset(
        x, nil, nil, nil, feature_names=["x"], target_names=["y"], description=["dummy"]
    )

    class DummyModel(SupervisedModel):
        """Under this model each data point receives a score of index / max,
        assuming that the values of training samples match their indices."""

        def __init__(self, data: Dataset):
            self.m = max(data.x_train)
            self.utility = 0

        def fit(self, x: ndarray, y: ndarray):
            self.utility = np.sum(x) / self.m

        def predict(self, x: ndarray) -> ndarray:
            return x

        def score(self, x: ndarray, y: ndarray) -> float:
            return self.utility

    return Utility(DummyModel(data), data, scoring=None, enable_cache=False)


@pytest.fixture(scope="session")
def linear_utility(linear_dataset):
    return Utility(LinearRegression(), data=linear_dataset, scoring=scoring)


@pytest.fixture(scope="function")
def exact_shapley(num_samples):
    """Scores are i/n, so v(i) = 1/n! Σ_π [U(S^π + {i}) - U(S^π)] = i/n"""
    u = dummy_utility(num_samples)
    exact_values = OrderedDict(
        {i: i / float(max(u.data.x_train)) for i in u.data.indices}
    )
    return u, exact_values


class TolerateErrors:
    """A context manager to swallow errors up to a certain threshold.
    Use to test (ε,δ)-approximations.
    """

    def __init__(
        self, max_errors: int, exception_cls: Type[BaseException] = AssertionError
    ):
        self.max_errors = max_errors
        self.Exception = exception_cls
        self.error_count = 0

    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            self.error_count += 1
        if self.error_count > self.max_errors:
            raise self.Exception(
                f"Maximum number of {self.max_errors} error(s) reached"
            )
        return True


def check_total_value(u: Utility, values: OrderedDict, atol: float = 1e-6):
    """Checks absolute distance between total and added values.
    Shapley value is supposed to fulfill the total value axiom."""
    total_utility = u(u.data.indices)
    values = np.fromiter(values.values(), dtype=float, count=len(u.data))
    # We could want relative tolerances here if we didn't have the range of
    # the scorer.
    assert np.isclose(values.sum(), total_utility, atol=atol)


def check_exact(values: OrderedDict, exact_values: OrderedDict, atol: float = 1e-6):
    """Compares ranks and values."""

    k = list(values.keys())
    ek = list(exact_values.keys())

    assert np.all(k == ek)

    v = np.array(list(values.values()))
    ev = np.array(list(exact_values.values()))

    assert np.allclose(v, ev, atol=atol)


def check_rank_correlation(
    values: OrderedDict,
    exact_values: OrderedDict,
    k: int = None,
    threshold: float = 0.9,
):
    """Checks that the indices of `values` and `exact_values` follow the same
    order (by value), with some slack, using Spearman's correlation.

    Runs an assertion for testing.

    :param values: The values and indices to test
    :param exact_values: The ground truth
    :param k: Consider only these many, starting from the top.
    :param threshold: minimal value for spearman correlation for the test to
        succeed
    """
    # FIXME: estimate proper threshold for spearman
    if k is not None:
        raise NotImplementedError
    else:
        k = len(values)
    ranks = np.array(list(values.keys())[:k])
    ranks_exact = np.array(list(exact_values.keys())[:k])

    assert spearman(ranks, ranks_exact) >= threshold


# start_logging_server()
