"""Compare, tune and save the late-order model.

Run with:  python -m latedeliveries.train
"""

from pathlib import Path

import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer, precision_score
from sklearn.model_selection import StratifiedKFold, cross_validate

from latedeliveries.pipeline import SEED, build_pipeline, load_split

ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "reports"
METRICS = ["accuracy", "precision", "recall", "f1", "roc_auc"]
# The baseline never predicts "late", so its precision is 0/0. Report it as 0.
SCORING = {m: m for m in METRICS} | {
    "precision": make_scorer(precision_score, zero_division=0)
}


def candidates() -> dict:
    """The baseline and the real models we compare, each by a short name."""
    return {
        "baseline": DummyClassifier(strategy="most_frequent"),
        "logistic": LogisticRegression(max_iter=1000),
        "logistic_balanced": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "random_forest": RandomForestClassifier(min_samples_leaf=5, random_state=SEED),
        "gradient_boosting": GradientBoostingClassifier(random_state=SEED),
    }


def compare_models(X_train, y_train, folds) -> pd.DataFrame:
    """Mean and standard deviation of every metric, from cross-validation."""
    rows = []
    for name, model in candidates().items():
        results = cross_validate(
            build_pipeline(model), X_train, y_train, cv=folds, scoring=SCORING
        )
        row = {"model": name}
        for m in METRICS:
            scores = results["test_" + m]
            row[m + "_mean"] = round(scores.mean(), 3)
            row[m + "_std"] = round(scores.std(), 3)
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    X, y, X_train, X_test, y_train, y_test = load_split()
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    table = compare_models(X_train, y_train, folds)
    REPORTS.mkdir(exist_ok=True)
    table.to_csv(REPORTS / "cv_results.csv", index=False)
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
