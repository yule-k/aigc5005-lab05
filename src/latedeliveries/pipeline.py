"""Feature lists, the train/test split and the shared preprocessing pipeline."""

import pandas as pd
from sklearn.model_selection import train_test_split

RANDOM_STATE = 42
TEST_SIZE = 0.25
TARGET = "late"

# Only columns whose value is known at the moment the order is placed.
NUMERIC_FEATURES = ["distance_km", "items", "prep_minutes", "courier_months"]
CATEGORY_FEATURES = ["hour", "weather", "day_type", "vehicle"]

# Left out on purpose; see README "Decisions".
EXCLUDED_COLUMNS = ["order_id", "customer_rating"]


def split_orders(frame: pd.DataFrame, random_state: int = RANDOM_STATE):
    """Return X_train, X_test, y_train, y_test with a stratified 75/25 split."""
    X = frame[NUMERIC_FEATURES + CATEGORY_FEATURES]
    y = frame[TARGET]
    return train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=random_state
    )