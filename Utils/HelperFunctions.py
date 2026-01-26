import numpy as np
import pandas as pd
from pandas import DataFrame
from typing import Any
import re

from Utils import StringHelperFunctions, Styling

def get_unique_number_patterns(series):
    """
    Extract all unique number-length patterns from a string Series.

    Parameters
    ----------
    series : pd.Series
        Series containing string values

    Returns
    -------
    list of str
        Unique patterns in the order of occurrence
    """
    # transform each string to number-length pattern
    transformed = series.astype(str).apply(
        lambda x: re.sub(r"\d+", lambda m: str(len(m.group())), x)
    )

    # unique patterns, in order of appearance
    seen = set()
    unique_patterns = []
    for pat in transformed:
        if pat not in seen:
            seen.add(pat)
            unique_patterns.append(pat)

    return unique_patterns

def clean_feature_names(columns):
    """
    Cleans column names (camel to snake, replace space with _,

    Parameters
    ----------
    columns : array of column names
    """
    cleaned = []
    for col in columns:
        col = col.strip()
        col = col.replace(" ", "_")
        col = StringHelperFunctions.camel_to_snake(col)
        cleaned.append(col)
    return cleaned

def _aggregate_df(df: DataFrame, feature_mask: list[str] | None, target_feature: list[Any] | None,
                 target_feature_name: str | None) -> tuple[Any, Any, str]:
    has_target_feature = target_feature_name is not None or target_feature is not None
    df = df.copy()
    if feature_mask is not None:
        if target_feature_name is not None:
            feature_mask.append(target_feature_name)
        df = df.loc[:, df.columns.intersection(feature_mask)]

    if has_target_feature:
        if target_feature_name is None:
            target_feature_name = "Target"
        if target_feature is not None:
            df[target_feature_name] = target_feature
    return df, has_target_feature, target_feature_name

def validate_pipeline(pipeline, X_train, X_test, pipeline_name, verbose = True):
    X_train_transformed = pipeline.fit_transform(X_train)
    X_test_transformed = pipeline.transform(X_test)

    print(f"{pipeline_name}: {X_train_transformed.shape}")
    train_nan_sum = np.isnan(X_train_transformed).sum()
    test_nan_sum = np.isnan(X_test_transformed).sum()

    if verbose:
        if train_nan_sum > 0:
            print(f"NaNs in X_train: {Styling.RED}{train_nan_sum}{Styling.RES}")

        if test_nan_sum > 0:
            print(f"NaNs in X_test: {Styling.RED}{test_nan_sum}{Styling.RES}")

        if train_nan_sum == 0 and test_nan_sum == 0:
            print(f"{Styling.GRE}no NaNs detected{Styling.RES}")

        print()



    return pd.DataFrame(X_train_transformed, columns=X_train.columns, index=X_train.index), pd.DataFrame(X_test_transformed, columns=X_test.columns, index=X_test.index)
