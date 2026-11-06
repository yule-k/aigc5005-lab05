"""Feature lists, the split and the preprocessing pipeline used by every model."""

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from latedeliveries.data import make_orders

SEED = 42
TARGET = "late"
# Known at the moment the order is placed.
NUMERIC = ["distance_km", "items", "prep_minutes", "courier_months"]
CATEGORICAL = ["hour", "weather", "day_type", "vehicle"]
# Left out on purpose: see "Decisions" in the README.
EXCLUDED = ["order_id", "customer_rating"]


def build_pipeline(model) -> Pipeline:
    """Imputing, scaling, encoding and the given model in one object."""
    preprocess = ColumnTransformer(
        [
            (
                "num",
                make_pipeline(SimpleImputer(strategy="median"), StandardScaler()),
                NUMERIC,
            ),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL,
            ),
        ]
    )
    return Pipeline([("prep", preprocess), ("model", model)])


def load_split(seed: int = SEED):
    """Return X, y and the stratified 75/25 split used throughout the lab."""
    orders = make_orders()
    X, y = orders[NUMERIC + CATEGORICAL], orders[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=seed
    )
    return X, y, X_train, X_test, y_train, y_test
