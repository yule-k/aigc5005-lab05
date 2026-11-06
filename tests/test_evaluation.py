"""Tests for evaluation.py, on examples small enough to check by hand."""

import pytest

from latedeliveries.evaluation import choose_threshold, evaluate, order_cost


def test_cost_of_a_small_example():
    y_true = [1, 1, 0, 0, 1, 0]
    flagged = [True, False, True, False, False, False]
    # two vouchers (2 x 4) and two missed late orders (2 x 15)
    assert order_cost(y_true, flagged) == 2 * 4 + 2 * 15


@pytest.mark.parametrize(
    ("flagged", "expected"),
    [
        ([False, False, False, False], 2 * 15),  # flag nobody: both late orders missed
        ([True, True, True, True], 4 * 4),  # flag everybody: four vouchers
    ],
)
def test_cost_of_the_two_extreme_policies(flagged, expected):
    y_true = [1, 0, 1, 0]
    assert order_cost(y_true, flagged) == expected


def test_threshold_can_choose_to_flag_nobody():
    # no order is late, so every voucher is wasted: flag as few as possible.
    # only 0.95 is above every probability, so it flags nobody and costs $0.
    y_true = [0, 0, 0]
    probability = [0.20, 0.50, 0.90]
    assert choose_threshold(y_true, probability) == 0.95


def test_threshold_can_choose_to_flag_everybody():
    # every order is late, so missing one ($15) costs more than a voucher ($4).
    # 0.05 flags all three: 3 x 4 = 12. 0.10 misses the first: 2 x 4 + 15 = 23.
    y_true = [1, 1, 1]
    probability = [0.06, 0.50, 0.90]
    assert choose_threshold(y_true, probability) == 0.05


def test_evaluate_reports_metrics_at_the_threshold():
    y_true = [1, 0, 1, 0]
    probability = [0.9, 0.6, 0.3, 0.1]
    # at 0.5: flagged = [1, 1, 0, 0] -> TN=1, FP=1, FN=1, TP=1
    scores = evaluate(y_true, probability, threshold=0.5)
    assert scores["confusion_matrix"] == [[1, 1], [1, 1]]
    assert scores["precision"] == 0.5
    assert scores["recall"] == 0.5
