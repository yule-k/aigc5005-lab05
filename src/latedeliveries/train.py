"""Compare, tune and save the late-order model.

Run with:  python -m latedeliveries.train
"""

from pathlib import Path

import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer, precision_score
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    cross_val_predict,
    cross_validate,
)

from latedeliveries.evaluation import choose_threshold, cost_by_threshold
from latedeliveries.pipeline import SEED, build_pipeline, load_split

ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "reports"
METRICS = ["accuracy", "precision", "recall", "f1", "roc_auc"]
# The baseline never predicts "late", so its precision is 0/0. Report it as 0.
SCORING = {m: m for m in METRICS} | {
    "precision": make_scorer(precision_score, zero_division=0)
}
CHOSEN = "logistic_balanced"
GRID = {
    "model__C": [0.01, 0.1, 1, 10],  # less or more regularisation
    "model__l1_ratio": [0, 0.5, 1],  # 0 = L2 penalty, 1 = L1 penalty
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


def tune_model(X_train, y_train, folds) -> GridSearchCV:
    """Grid search on the chosen model. ROC-AUC, because the threshold comes later."""
    model = LogisticRegression(max_iter=5000, class_weight="balanced", solver="saga")
    search = GridSearchCV(
        build_pipeline(model), GRID, scoring="roc_auc", cv=folds, n_jobs=-1
    )
    return search.fit(X_train, y_train)


def main() -> None:
    X, y, X_train, X_test, y_train, y_test = load_split()
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    table = compare_models(X_train, y_train, folds)
    REPORTS.mkdir(exist_ok=True)
    table.to_csv(REPORTS / "cv_results.csv", index=False)
    print(table.to_string(index=False))

    search = tune_model(X_train, y_train, folds)
    untuned = table.set_index("model").loc[CHOSEN]
    spread = search.cv_results_["std_test_score"][search.best_index_]
    print("Best settings:", search.best_params_)
    print(f"Tuned ROC-AUC   {search.best_score_:.3f} +/- {spread:.3f}")
    print(f"Untuned ROC-AUC {untuned.roc_auc_mean:.3f} +/- {untuned.roc_auc_std:.3f}")

    probability = cross_val_predict(
        search.best_estimator_, X_train, y_train, cv=folds, method="predict_proba"
    )[:, 1]
    threshold = choose_threshold(y_train, probability)
    print(cost_by_threshold(y_train, probability).to_string(index=False))
    print("Threshold with the lowest out-of-fold cost:", threshold)


if __name__ == "__main__":
    main()
