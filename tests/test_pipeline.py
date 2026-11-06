import pytest

from latedeliveries.pipeline import (
    CATEGORY_FEATURES,
    EXCLUDED_COLUMNS,
    NUMERIC_FEATURES,
    TARGET,
    split_orders,
)


@pytest.mark.parametrize("column", EXCLUDED_COLUMNS + [TARGET])
def test_excluded_columns_are_not_features(column):
    assert column not in NUMERIC_FEATURES + CATEGORY_FEATURES


def test_split_is_stratified(orders):
    X_train, X_test, y_train, y_test = split_orders(orders)
    assert len(X_train) == 2250 and len(X_test) == 750
    assert abs(y_train.mean() - y_test.mean()) < 0.01


def test_split_is_repeatable(orders):
    first = split_orders(orders)[0]
    second = split_orders(orders)[0]
    assert first.index.equals(second.index)
