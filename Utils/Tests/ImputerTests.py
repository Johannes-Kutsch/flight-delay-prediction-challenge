import math

import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from Utils.Imputers import PredictiveImputer
from Utils.Transformers import SequentialImputer

from sklearn.linear_model import LinearRegression


def test_simple():
    X = pd.DataFrame(
        {
            "A": [5.0, 8.0, np.nan, 20.0, np.nan],
            "B": [2.0, np.nan, 6.0, np.nan, 14.0],
        }
    )

    stages = [
        {
            "model": SimpleImputer(strategy="mean"),
            "column_mask": ["A"]
        },
        SimpleImputer(strategy="median"),
    ]

    pipeline = Pipeline([
        ("impute Features", SequentialImputer(stages)),
        ])

    X_out = pipeline.fit_transform(X)

    print(X_out.columns)

    check_columns(X_out, {"A", "B"})
    check_nan(X_out)

    expected_values = {
        "A": [5.0, 8.0, 11, 20.0, 11],
        "B": [2.0, 6.0, 6.0, 6.0, 14.0]
    }

    check_values(X_out, expected_values)

def test_missing_indicator():
    X = pd.DataFrame(
        {
            "A": [5.0, 8.0, np.nan, 20.0, np.nan],
            "B": [2.0, np.nan, 6.0, np.nan, 14.0],
        }
    )

    stages = [
        {
            "model": SimpleImputer(strategy="mean", add_indicator=True),
            "column_mask": ["A"]
        },
        SimpleImputer(strategy="median", add_indicator=True),
    ]

    pipeline = Pipeline([
        ("impute Features", SequentialImputer(stages)),
        ])

    X_out = pipeline.fit_transform(X)

    check_columns(X_out, {"A", "B", "missingindicator_A", "missingindicator_B"})
    check_nan(X_out)

    expected_values = {
        "A": [5.0, 8.0, 11, 20.0, 11],
        "B": [2.0, 6.0, 6.0, 6.0, 14.0],
        "missingindicator_A" : [0, 0, 1, 0, 1],
        "missingindicator_B": [0, 1, 0, 1, 0],
    }

    check_values(X_out, expected_values)

def test_duplicate_feature():
    X = pd.DataFrame(
        {
            "A": [5.0, 8.0, np.nan, 20.0, np.nan],
            "B": [2.0, np.nan, 6.0, np.nan, 14.0],
        }
    )

    stages = [
        {
            "model": SimpleImputer(strategy="mean"),
            "column_mask": ["A"],
            "new_column_suffix": "_filled",
        },
        {
            "model": SimpleImputer(strategy="median"),
            "column_mask": ["B"],
            "new_column_suffix": "_filled",
        },
    ]

    pipeline = Pipeline([
        ("impute Features", SequentialImputer(stages)),
        ])

    X_out = pipeline.fit_transform(X)
    expected_values = {
        "A": [5.0, 8.0, np.nan, 20.0, np.nan],
        "B": [2.0, np.nan, 6.0, np.nan, 14.0],
        "A_filled": [5.0, 8.0, 11, 20.0, 11],
        "B_filled": [2.0, 6.0, 6.0, 6.0, 14.0],
    }

    check_columns(X_out, {"A", "B", "A_filled", "B_filled"})
    check_values(X_out, expected_values)

def test_duplicate_feature_missing_indicator():
    X = pd.DataFrame(
        {
            "A": [5.0, 8.0, np.nan, 20.0, np.nan],
            "B": [2.0, np.nan, 6.0, np.nan, 14.0],
        }
    )

    stages = [
        {
            "model": SimpleImputer(strategy="mean", add_indicator=True),
            "column_mask": ["A"],
            "new_column_suffix": "_filled",
        },
        {
            "model": SimpleImputer(strategy="median", add_indicator=True),
            "column_mask": ["B"],
            "new_column_suffix": "_filled",
        },
    ]

    pipeline = Pipeline([
        ("impute Features", SequentialImputer(stages)),
        ])

    X_out = pipeline.fit_transform(X)
    expected_values = {
        "A": [5.0, 8.0, np.nan, 20.0, np.nan],
        "B": [2.0, np.nan, 6.0, np.nan, 14.0],
        "A_filled": [5.0, 8.0, 11, 20.0, 11],
        "B_filled": [2.0, 6.0, 6.0, 6.0, 14.0],
        "missingindicator_A_filled": [0, 0, 1, 0, 1],
        "missingindicator_B_filled": [0, 1, 0, 1, 0]
    }

    check_columns(X_out, {"A", "B", "A_filled", "B_filled", "missingindicator_A_filled", "missingindicator_B_filled"})
    check_values(X_out, expected_values)

def test_linear_regression():
    X = pd.DataFrame(
        {
            "A": [5.0, 7.0, np.nan, 11.0, np.nan],
            "B": [2.0, 4.0, 6.0, 8.0, 10.0],
        }
    )

    stages = [
        PredictiveImputer("A", LinearRegression()),
    ]

    pipeline = Pipeline([
        ("impute Features", SequentialImputer(stages)),
        ])

    X_out = pipeline.fit_transform(X)

    expected_values = {
        "A": [5.0, 7.0, 9.0, 11.0, 13.0],
    }

    check_columns(X_out, {"A", "B"})
    check_nan(X_out)
    check_values_tolerance(X_out, expected_values, 1e1)

def check_nan(X_out, cols = None):
    if cols is None:
        for col in X_out.columns:
            assert X_out[col].isna().sum() == 0, f"Column {col} still contains NaNs"
    else:
        for col in cols:
            assert X_out[col].isna().sum() == 0, f"Column {col} still contains NaNs"

def check_columns(X_out, expected_columns: set[str]):
    missing_cols = expected_columns - set(X_out.columns)
    assert not missing_cols, f"Missing expected columns: {missing_cols}, actual columns: {X_out.columns}"

def check_values(X_out, expected_values_by_feature):
    for feature, expected_values in expected_values_by_feature.items():
        actual_list = X_out[feature].tolist()
        assert len(actual_list) == len(expected_values), (
            f"Length of feature '{feature}' is wrong: {len(actual_list)} instead of {len(expected_values)}"
        )
        for i, (actual, expected) in enumerate(zip(actual_list, expected_values)):
            assert actual == expected or (math.isnan(expected) and math.isnan(actual)), (
                f"Feature '{feature}', Index {i}: Value {actual} != expected {expected}"
            )

def check_values_tolerance(X_out, expected_values_by_feature, tolerance = 2e1):
    for feature, expected_values in expected_values_by_feature.items():
        actual_list = X_out[feature].tolist()
        assert len(actual_list) == len(expected_values), (
            f"Length of feature '{feature}' is wrong: {len(actual_list)} instead of {len(expected_values)}"
        )
        for i, (actual, expected) in enumerate(zip(actual_list, expected_values)):
            assert abs(actual - expected) < tolerance or (math.isnan(expected) and math.isnan(actual)), (
                f"Feature '{feature}', Index {i}: Value {actual} != expected {expected}"
            )