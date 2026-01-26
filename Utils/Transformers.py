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
    def __init__(self, column, mapping, new_column=None, drop_old_column = True):
        """
        column: Name of feature
        mapping: dict i.e. {"male": True, "female": False}
        new_column: Name der neuen Spalte, wenn None -> Original überschreiben
        """
        self.column = column
        self.mapping = mapping
        self.new_column = new_column
        self.drop_old_column = drop_old_column

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if self.column not in X.columns:
            raise ValueError(f"Column '{self.column}' is no Feature inside this DataFrame")

        X = X.copy()
        X[self.column] = X[self.column].map(self.mapping)

        if self.new_column and self.drop_old_column:
            X[self.new_column] = X[self.column]
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