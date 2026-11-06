"""Tests for saving and loading the whole pipeline with joblib."""

import joblib


def test_saved_model_predicts_the_same_after_reloading(tmp_path, split, fitted):
    X_train = split[2]
    path = tmp_path / "late_orders.joblib"
    joblib.dump(fitted, path)
    loaded = joblib.load(path)
    assert (loaded.predict_proba(X_train) == fitted.predict_proba(X_train)).all()
