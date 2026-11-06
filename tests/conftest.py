"""Fixtures shared by every test file."""

import pytest
from sklearn.linear_model import LogisticRegression

from latedeliveries.pipeline import build_pipeline, load_split


@pytest.fixture(scope="session")
def split():
    return load_split()


@pytest.fixture(scope="session")
def fitted(split):
    X, y, X_train, X_test, y_train, y_test = split
    return build_pipeline(LogisticRegression(max_iter=1000)).fit(X_train, y_train)
