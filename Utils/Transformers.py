import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd

from Utils import HelperFunctions


class FeatureNameCleaner(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X.columns = HelperFunctions.clean_feature_names(X.columns)
        return X

class FeatureRenameTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, rename_map):
        """
        rename_map: dict mapping old feature names to new feature names
                    Example: {"BloodPressure": "blood_pressure",
                              "SkinThickness": "skin_thickness"}
        """
        self.rename_map = rename_map

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        missing_cols = set(self.rename_map.keys()) - set(X.columns)
        if missing_cols:
            raise ValueError(
                f"The following columns are not present in the DataFrame: {missing_cols}"
            )

        X = X.rename(columns=self.rename_map)
        return X

class TypeCastTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, column, target_dtype):
        self.column = column
        self.target_dtype = target_dtype

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if self.column not in X.columns:
            raise ValueError(f"Column '{self.column}' is no Feature inside this DataFrame")

        X = X.copy()
        X[self.column] = X[self.column].astype(self.target_dtype)

        return X

class RemapTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, column, mapping, new_column=None, drop_old_column=True):
        """
        Transformer to remap values of a DataFrame column according to a dictionary.
        Values not present in the mapping are kept unchanged.

        Parameters
        ----------
        column : str
            Name of the column to transform.
        mapping : dict
            Dictionary of old_value -> new_value.
        new_column : str or None
            Name of the new column. If None, overwrite original column.
        drop_old_column : bool
            Whether to drop the old column if new_column is set.
        """
        self.column = column
        self.mapping = mapping
        self.new_column = new_column
        self.drop_old_column = drop_old_column

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if self.column not in X.columns:
            raise ValueError(f"Column '{self.column}' is not present in the DataFrame")

        X = X.copy()

        target_col = self.new_column if self.new_column else self.column

        X[target_col] = X[self.column].replace(self.mapping)

        if self.new_column and self.drop_old_column:
            X = X.drop(columns=[self.column])

        return X

class ThresholdToNaNTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, thresholds, mode="min", drop_old_column=True, new_column_suffix="_cleaned"):
        """
        thresholds: dict mapping feature names to threshold values.
                    Example: {"age": 18, "income": 1000}
        mode: "min" -> values <= threshold set to NaN
              "max" -> values >= threshold set to NaN
        new_column_suffix: string to append to feature names for new columns.
        drop_old_column: if True, original columns are removed after transformation.
        """
        self.thresholds = thresholds
        self.mode = mode
        self.new_column_suffix = new_column_suffix
        self.drop_old_column = drop_old_column

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        for col, threshold in self.thresholds.items():
            if col not in X.columns:
                raise ValueError(f"Column '{col}' is not a feature inside this DataFrame")

            if self.mode == "min":
                transformed = X[col].where(X[col] > threshold, np.nan)
            elif self.mode == "max":
                transformed = X[col].where(X[col] < threshold, np.nan)
            else:
                raise ValueError("mode must be 'min' or 'max'")

            if self.drop_old_column:
                X[col] = transformed
            else:
                X[col + self.new_column_suffix] = transformed



        return X

class DropFeaturesTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, columns: list[str]):
        """
        columns: array i.e. ['name', 'cabin', 'ticket', 'sex']
        """
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        for column in self.columns:
            if column not in X.columns:
                raise ValueError(f"Column '{column}' is no Feature inside this DataFrame")

            X = X.drop(columns=[column])
        return X

class FeatureMultiplier(BaseEstimator, TransformerMixin):
    """
    Multiplies the value of all features for each row and creates a new feature containing the result
    columns: array i.e. ['name', 'cabin', 'ticket', 'sex']
    """
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        product = X[self.columns[0]].copy()
        for col in self.columns[1:]:
            product *= X[col]
        X["__".join(self.columns)] = product

        return X

class DropNaNRowsTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, features=None):
        """
        features: list of feature names or None.
                  - If list: rows with NaN in any of these features will be dropped.
                  - If None: rows with NaN in any column will be dropped.
        """
        self.features = features

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        if self.features is None:
            return X.dropna()

        missing_cols = set(self.features) - set(X.columns)
        if missing_cols:
            raise ValueError(
                f"The following columns are not present in the DataFrame: {missing_cols}"
            )

        return X.dropna(subset=self.features)

class SequentialImputer(BaseEstimator, TransformerMixin):
    """
    Stage-wise imputer.
    Applies each imputer sequentially and can add a suffix for filled columns.
    Returns a DataFrame with proper column names.
    """

    def __init__(self, stages):
        """
        stages : list of dicts
            [
                {
                    "model": SimpleImputer(...),
                    "column_mask": ["A", "B"],
                    "new_column_suffix": "_filled"  # optional
                },
                ...
            ]
        """
        self.stages = stages

        self.fitted_stages_ = None

    def fit(self, X, y=None):
        self.stages = self.aggregate_stages(self.stages)

        X_work = X.copy()
        self.fitted_stages_ = []

        for stage in self.stages:
            model = stage["model"]
            cols = stage.get("column_mask", X_work.columns)
            suffix = stage.get("new_column_suffix", "")

            model.fit(X_work[cols])
            X_work = self._apply_stage(X_work, model, cols, suffix)

            self.fitted_stages_.append((model, cols, suffix))

        return self

    @staticmethod
    def aggregate_stages(stages):
        normalized_stages = []
        for stage in stages:
            if isinstance(stage, dict):
                normalized_stages.append(stage)
            else:
                normalized_stages.append({"model": stage})
        return normalized_stages

    def transform(self, X):
        X_work = X.copy()

        for model, cols, suffix in self.fitted_stages_:
            X_work = self._apply_stage(X_work, model, cols, suffix)

        return X_work

    @staticmethod
    def _apply_stage(X, model, cols, suffix=""):
        transformed = model.transform(X[cols])
        if transformed.ndim == 1:
            transformed = transformed.reshape(-1, 1)

        transformed_feature_names = model.get_feature_names_out(cols)

        df_transformed = pd.DataFrame(transformed, columns=transformed_feature_names, index=X.index)

        df_transformed = SequentialImputer._drop_unchanged(df_transformed, X)
        if suffix:
            df_transformed.columns = df_transformed.columns + suffix
            X_out = pd.concat([X, df_transformed], axis=1)
        else:
            X_out = X.copy()
            for col in df_transformed:
                X_out[col] = df_transformed[col]

        return X_out

    @staticmethod
    def _drop_unchanged(df_transformed, X, prefix="missing_indicator_"):
        cols_to_drop = []

        for col in df_transformed.columns:
            if col in X.columns:
                if X[col].isna().any():
                    continue
                if df_transformed[col].equals(X[col]):
                    cols_to_drop.append(col)

                    indicator_col = f"{prefix}{col}"
                    if indicator_col in df_transformed.columns:
                        cols_to_drop.append(indicator_col)

        df_transformed = df_transformed.drop(columns=cols_to_drop)
        return df_transformed

class DFMerger(BaseEstimator, TransformerMixin):
    """
    Transformer for merging a DataFrame with another DataFrame within a scikit-learn pipeline.

    Parameters
    ----------
    df_to_merge : pd.DataFrame
        The DataFrame to merge into the input X.
    left_on : str
        Column name in X to join on.
    right_on : str
        Column name in df_to_merge to join on.
    how : str, default="left"
        Type of merge: 'left', 'right', 'inner', 'outer'.
    prefix : str, default=None
        Optional prefix to prepend to all columns from df_to_merge in the merged DataFrame.
    """

    def __init__(self, df_to_merge, left_on, right_on, how="left", prefix=None):
        self.df_to_merge = df_to_merge.copy()
        self.left_on = left_on
        self.right_on = right_on
        self.how = how
        self.prefix = prefix

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        df_to_merge = self.df_to_merge.copy()

        if self.prefix:
            df_to_merge = df_to_merge.add_prefix(self.prefix)

        right_on = self.right_on
        if self.prefix:
            right_on = self.prefix + self.right_on

        X_merged = X.merge(df_to_merge, how=self.how, left_on=self.left_on, right_on=right_on)

        return X_merged

class DistanceTransformer(BaseEstimator, TransformerMixin):
    """
    Compute the great-circle (Haversine) distance between two lat/long coordinates.

    Parameters
    ----------
    lat_1 : str
        Name of the first latitude column.
    lon_1 : str
        Name of the first longitude column.
    lat_1 : str
        Name of the second latitude column.
    lon_2 : str
        Name of the second longitude column.
    new_feature_name : str, default="distance"
        Name of the generated distance column.
    """

    def __init__(self, lat_1, lon_1,
                 lat_2, lon_2,
                 new_feature_name="distance_km"):
        self.lat_1 = lat_1
        self.lon_1 = lon_1
        self.lat_2 = lat_2
        self.lon_2 = lon_2
        self.column_name = new_feature_name

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        dep_lat_rad = np.radians(X[self.lat_1])
        dep_lon_rad = np.radians(X[self.lon_1])
        arr_lat_rad = np.radians(X[self.lat_2])
        arr_lon_rad = np.radians(X[self.lon_2])

        lat_distance = arr_lat_rad - dep_lat_rad
        lon_distance = arr_lon_rad - dep_lon_rad

        a = np.sin(lat_distance / 2) ** 2 + np.cos(dep_lat_rad) * np.cos(arr_lat_rad) * np.sin(lon_distance / 2) ** 2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
        distance = 6371  * c

        X[self.column_name] = distance
        return X

class TypeCastDatetimeTransformer(BaseEstimator, TransformerMixin):
    """
    Transformer to cast a column to datetime, with optional custom format.
    """

    def __init__(self, column, datetime_format=None):
        self.column = column
        self.datetime_format = datetime_format

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X[self.column] = pd.to_datetime(
            X[self.column],
            format=self.datetime_format,
            errors="raise"
        )
        return X

class DatetimeDifferenceTransformer(BaseEstimator, TransformerMixin):
    """
    Compute the time difference between two datetime columns.

    Parameters
    ----------
    start_column : str
        Name of the start datetime column.
    end_column : str
        Name of the end datetime column.
    new_column : str
        Name of the output feature.
    unit : str, default="hours"
        Unit of the time difference.
        One of: "seconds", "minutes", "hours", "days".
    """

    def __init__(self, start_column, end_column, new_column, unit="hours"):
        self.start_column = start_column
        self.end_column = end_column
        self.new_column = new_column
        self.unit = unit

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        if self.start_column not in X.columns or self.end_column not in X.columns:
            raise ValueError("Start or end datetime column not found in DataFrame")

        delta = X[self.end_column] - X[self.start_column]

        if self.unit == "seconds":
            X[self.new_column] = delta.dt.total_seconds()
        elif self.unit == "minutes":
            X[self.new_column] = delta.dt.total_seconds() / 60
        elif self.unit == "hours":
            X[self.new_column] = delta.dt.total_seconds() / 3600
        elif self.unit == "days":
            X[self.new_column] = delta.dt.total_seconds() / 86400
        else:
            raise ValueError(f"Unsupported unit: {self.unit}")

        return X

class EqualityFlagTransformer(BaseEstimator, TransformerMixin):
    """
    Create a binary flag indicating whether two feature columns have identical values.

    Parameters
    ----------
    column_a : str
        Name of the first column.
    column_b : str
        Name of the second column.
    new_column : str
        Name of the output flag column.
    """

    def __init__(self, column_a, column_b, new_column,
                 true_value=1, false_value=0):
        self.column_a = column_a
        self.column_b = column_b
        self.new_column = new_column

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        if self.column_a not in X.columns or self.column_b not in X.columns:
            raise ValueError("One or both columns not found in DataFrame")

        X[self.new_column] = (
            X[self.column_a]
            .eq(X[self.column_b])
            .fillna(False)
        )

        return X