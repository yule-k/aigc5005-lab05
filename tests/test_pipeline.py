"""Tests for pipeline.py. Run with:  python -m pytest -v"""

import numpy as np
import pandas as pd
import pytest

from latedeliveries.pipeline import CATEGORICAL, EXCLUDED, NUMERIC, TARGET, load_split


@pytest.mark.parametrize("column", EXCLUDED + [TARGET])
def test_feature_lists_leave_out(column):
    assert column not in NUMERIC + CATEGORICAL


def test_split_is_stratified(split):
    X, y, X_train, X_test, y_train, y_test = split
    assert (len(X_train), len(X_test)) == (2250, 750)
    assert y_train.mean() == pytest.approx(y_test.mean(), abs=0.01)


def test_split_is_repeatable():
    first, second = load_split(), load_split()
    pd.testing.assert_frame_equal(first[2], second[2])


def test_pipeline_gives_probabilities_between_0_and_1(split, fitted):
    X_train = split[2]
    probabilities = fitted.predict_proba(X_train)
    assert probabilities.shape == (len(X_train), 2)
    assert ((probabilities >= 0) & (probabilities <= 1)).all()
    assert probabilities.sum(axis=1) == pytest.approx(1.0)


def test_pipeline_handles_missing_value_and_unseen_category(fitted):
    new = pd.DataFrame(
        [
            {
                "distance_km": 3.2,
                "items": 2,
                "prep_minutes": np.nan,
                "courier_months": 12,
                "hour": 19,
                "weather": "hail",
                "day_type": "weekday",
                "vehicle": "bike",
            }
        ]
    )
    assert fitted.predict_proba(new).shape == (1, 2)
