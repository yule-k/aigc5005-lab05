import pytest

from latedeliveries.data import make_orders


@pytest.fixture(scope="session")
def orders():
    return make_orders()
