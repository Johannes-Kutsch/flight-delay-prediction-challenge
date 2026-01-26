from typing import Any

import pandas as pd
from pandas import DataFrame
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression


def impute_missing_values(df: DataFrame, features, imputer, output_mask) -> Any:
    imputed = df[features].copy()

    imputed = pd.DataFrame(
        imputer.transform(imputed),
        columns=features,
        index=df.index
    )

    return imputed.loc[output_mask]


class PredictiveImputer(BaseEstimator, TransformerMixin):
    """
    Predictive imputer for a single column.

    This transformer follows the interface of `SimpleImputer` (fit -> transform),
    but allows missing values in a target column to be predicted using a regression model.

    Parameters
    ----------
    model : object, default=LinearRegression()
        Regression model with fit(X, y) and predict(X).

    target_column : str
        Column to be imputed.

    predictor_features : list of str, optional
        Columns used as predictors. If None, all non target columns are used.

    predictor_imputer : Object, default = SimpleImputer(strategy="mean")
        SimpleImputer (or similar) for missing predictor values.
    """

    def __init__(self, target_column: str, model=None, predictor_imputer=None, predictor_features: list[str]=None):
        self.target_column = target_column
        self.model = LinearRegression() if model is None else model
        self.predictor_imputer = SimpleImputer(strategy="mean") if predictor_imputer is None else predictor_imputer
        self.predictor_features = predictor_features

        self.target_column_index_ = None
        self.predictor_features_indices_ = None
        self._imputed_predictor_indices_ = None

    def fit(self, X):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        self.predictor_features_indices_, self.target_column_index_ = get_feature_indices(X, self.target_column, self.predictor_features)

        target_mask = pd.notna(X.iloc[:, self.target_column_index_])
        if not target_mask.values.any():
            raise ValueError(f"No Valid Rows to train feature {self.target_column}!")

        X_train = X.iloc[target_mask.values, self.predictor_features_indices_].copy()

        self._imputed_predictor_indices_ = _fit_predictor_imputer(X, self.predictor_imputer, self.predictor_features_indices_, self.target_column)


        predictor_nan_mask = X_train.isna().any(axis=1)
        if predictor_nan_mask.any():
            X_train.loc[predictor_nan_mask] = self.predictor_imputer.transform(
                X_train.loc[predictor_nan_mask]
            )

        y_train = X.iloc[target_mask.values, self.target_column_index_]

        self.model.fit(X_train, y_train)
        return self

    def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        X_out = X.copy()

        target_mask = X_out.iloc[:, self.target_column_index_].isna()
        if not target_mask.any():
            return X_out

        X_pred = X_out.loc[target_mask, X_out.columns[self.predictor_features_indices_]]

        if self._imputed_predictor_indices_:
            predictor_nan_mask = X_pred[self._imputed_predictor_indices_].isna().any(axis=1)

            if predictor_nan_mask.any():
                X_pred.loc[predictor_nan_mask, self._imputed_predictor_indices_] = (
                    self.predictor_imputer.transform(
                        X_pred.loc[predictor_nan_mask, self._imputed_predictor_indices_]
                    )
                )

        predicted_values = self.model.predict(X_pred)

        if predicted_values.ndim != 1:
                raise ValueError(f"Too many Columns after imputation for {self.target_column} ({predicted_values.ndim})!")

        X_out.loc[target_mask, X_out.columns[self.target_column_index_]] = predicted_values

        return X_out

    def get_feature_names_out(self, input_features=None):
        """
        Return output feature names.

        PredictiveImputer does not change the number or order of features.
        """

        if input_features is None:
            raise ValueError("input_features must be provided")
        return list(input_features)

def _fit_predictor_imputer(X, imputer, predictor_feature_indices, imputer_name):
    """
    Fit the predictor imputer on predictor features.

    Only validates feasibility; does not modify X.
    """
    X_predictors = X.iloc[:, predictor_feature_indices]

    if X_predictors.shape[1] == 0:
        raise ValueError(
            f"No predictor columns available to fit imputer for {imputer_name}!"
        )

    _imputed_predictor_indices = X_predictors.columns[X_predictors.isna().any()]

    if len(_imputed_predictor_indices) == 0:
        return []

    X_imputed_predictors = X_predictors[_imputed_predictor_indices]

    if X_imputed_predictors.dropna(how="all").shape[0] == 0:
        raise ValueError(
            f"All predictor rows are NaN for feature {imputer_name}, cannot fit imputer!"
        )

    imputer.fit(X_imputed_predictors)

    return list(_imputed_predictor_indices)

def get_feature_indices(X, target_column: str, predictor_features: list[str] = None):
    """
    Determines the indices of predictor columns and the target column.

    Args:
        X (pd.DataFrame): Input dataframe containing features and target.
        target_column (str): Column to be imputed.
        predictor_features (list of str or None): Features used as predictors for imputation.

    Returns:
        predictor_indices (list of int): Indices of the predictor columns.
        target_index (int): Index of the target column.

    """
    columns = X.columns


    if predictor_features is None:
        target_index = columns.get_loc(target_column)
        predictor_indices = [i for i, c in enumerate(columns) if c != target_column]
    else:
        predictor_indices = [columns.get_loc(c) for c in predictor_features]

    return predictor_indices, target_index