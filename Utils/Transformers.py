import numpy as np
import pytz
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.mixture import BayesianGaussianMixture
from sklearn.neighbors import BallTree
from sklearn.preprocessing import StandardScaler, RobustScaler

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
    """
    Transformer to cast multiple columns to specified dtypes.

    Parameters
    ----------
    dtype_mapping : dict
        Dictionary mapping column names to target dtypes, e.g.,
        {'col1': 'float32', 'col2': 'category'}
    """

    def __init__(self, dtype_mapping: dict):
        self.dtype_mapping = dtype_mapping

    def fit(self, X, y=None):
        missing_cols = [col for col in self.dtype_mapping if col not in X.columns]
        if missing_cols:
            raise ValueError(f"Columns {missing_cols} are not in the DataFrame")
        return self

    def transform(self, X):
        X = X.copy()
        for col, target_dtype in self.dtype_mapping.items():
            X[col] = X[col].astype(target_dtype)
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
    Transformer to cast a column to datetime, with optional custom format and optional per-row timezone conversion.

    Parameters
    ----------
    column : str
        Name of the column to cast to datetime.
    datetime_format : str, optional
        Optional datetime format for parsing.
    tz_column : str, optional
        Name of the column containing the timezone strings (e.g., 'Europe/Paris').
        If provided, the datetime will be converted to the corresponding local timezone per row.
    """

    def __init__(self, column, datetime_format=None, tz_column=None):
        self.column = column
        self.datetime_format = datetime_format
        self.tz_column = tz_column

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        X[self.column] = pd.to_datetime(X[self.column], format=self.datetime_format, errors="raise")

        if self.tz_column is not None:
            X[self.column] = X[self.column].dt.tz_localize('UTC')

            for tz in X[self.tz_column].unique():
                mask = X[self.tz_column] == tz
                X.loc[mask, self.column] = X.loc[mask, self.column].dt.tz_convert(pytz.timezone(tz))

        return X

class LocalizeDatetimePerRowTransformer(BaseEstimator, TransformerMixin):
    """
    Create a new tz-aware datetime column based on a tz-naive datetime column
    and a per-row timezone column.

    Parameters
    ----------
    datetime_column : str
        Name of the tz-naive datetime column.
    tz_column : str
        Name of the column with the timezone strings (e.g., 'Europe/Paris').
    new_column : str
        Name of the new tz-aware datetime column.
    """

    def __init__(self, datetime_column, tz_column, new_column):
        self.datetime_column = datetime_column
        self.tz_column = tz_column
        self.new_column = new_column

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        X[self.new_column] = pd.concat([
            X.loc[X[self.tz_column] == tz, self.datetime_column]
            .dt.tz_localize(tz).dt.tz_convert("UTC")
            for tz in X[self.tz_column].unique()
        ]).sort_index()

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

    def __init__(self, column_a, column_b, new_column, invert_output=False):
        self.column_a = column_a
        self.column_b = column_b
        self.new_column = new_column
        self.invert_output = invert_output

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        if self.column_a not in X.columns or self.column_b not in X.columns:
            raise ValueError("One or both columns not found in DataFrame")

        X[self.new_column] = X[self.column_a].eq(X[self.column_b]).fillna(False)

        if self.invert_output:
            X[self.new_column] = ~X[self.new_column]

        return X

class RegexFlagTransformer(BaseEstimator, TransformerMixin):
    """
    A scikit-learn compatible transformer that creates a binary flag column
    based on whether values in a specified column match a given regex pattern.

    Parameters
    ----------
    target_column : str
        The name of the column to apply the regex to.
    pattern : str
        The regex pattern to match against the target column.
    new_column : str
        The name of the new column to store the flag (0/1).
    negate : bool, default=False
        If True, inverts the flag (e.g., for marking "pseudo" instead of "real").
    """

    def __init__(self, target_column, pattern, new_column, strip_string=False, negate=False):
        self.target_column = target_column
        self.pattern = pattern
        self.new_column = new_column
        self.strip_string = strip_string
        self.negate = negate

    def fit(self, X, y=None):
        """
        Fit method, does nothing but required for sklearn compatibility.
        """
        return self

    def transform(self, X):
        """
        Apply the regex match to the target column and create the new flag column.

        Parameters
        ----------
        X : pd.DataFrame
            Input dataframe.

        Returns
        -------
        X_transformed : pd.DataFrame
            Copy of the input dataframe with an additional flag column.
        """
        X = X.copy()
        flag = self._apply_pattern(X[self.target_column], self.pattern, strip=self.strip_string)

        if self.negate:
            flag = ~flag
        X[self.new_column] = flag
        return X

    @staticmethod
    def _apply_pattern(series, pattern, strip=True):
        if strip:
            series = series.str.strip()
        return series.str.contains(pattern, na=False)

class ConditionalValueUpdater(BaseEstimator, TransformerMixin):
    """
    Sets or updates a column based on a condition applied to another column.
    Existing values are preserved if the condition is not met.

    Parameters
    ----------
    condition_column : str
        Name of the column to check the condition against.
    condition_values : list or any
        The value(s) in `condition_column` that trigger the assignment.
    target_column : str
        Name of the column to create or update.
    new_value : any
        The value to set in `target_column` when the condition is met.
    """
    def __init__(self, condition_column, condition_values, target_column, new_value):
        self.condition_column = condition_column
        self.condition_values = condition_values
        self.target_column = target_column
        self.new_value = new_value

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        cond_vals = self.condition_values if isinstance(self.condition_values, list) else [self.condition_values]

        mask = X[self.condition_column].isin(cond_vals)

        X.loc[mask, self.target_column] = self.new_value

        return X

class DateTimeFeatureExtractor(BaseEstimator, TransformerMixin):
    """
    Extracts comprehensive date/time features from a datetime column.
    'month' and 'weekday' can optionally be returned as categorical strings.

    Parameters
    ----------
    datetime_column : str
        Name of the datetime column to extract features from.
    features : list of str, optional
        List of features to extract. Possible values:
        'year', 'month', 'day', 'weekday', 'hour', 'minute', 'second',
        'quarter', 'dayofyear', 'is_weekend'.
        Default is all features.
    month_as_category : bool, default=True
        If True, month numbers are converted to string names (e.g., 'Jan', 'Feb').
    weekday_as_category : bool, default=True
        If True, weekdays are converted to string names (e.g., 'Mon', 'Tue').
    """

    ALL_FEATURES = ['year', 'month', 'day', 'weekday', 'hour', 'minute', 'second',
                    'quarter', 'dayofyear', 'is_weekend']

    MONTH_NAMES = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    WEEKDAY_NAMES = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']

    def __init__(self, datetime_column, features=None, month_as_category=True, weekday_as_category=True):
        self.datetime_column = datetime_column
        self.features = features or self.ALL_FEATURES
        self.month_as_category = month_as_category
        self.weekday_as_category = weekday_as_category

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        dt = X[self.datetime_column]

        if 'year' in self.features:
            X[f'{self.datetime_column}_year'] = dt.dt.year
        if 'month' in self.features:
            if self.month_as_category:
                X[f'{self.datetime_column}_month'] = dt.dt.month.apply(lambda x: self.MONTH_NAMES[x - 1])
            else:
                X[f'{self.datetime_column}_month'] = dt.dt.month
        if 'day' in self.features:
            X[f'{self.datetime_column}_day'] = dt.dt.day
        if 'weekday' in self.features:
            if self.weekday_as_category:
                X[f'{self.datetime_column}_weekday'] = dt.dt.weekday.apply(lambda x: self.WEEKDAY_NAMES[x])
            else:
                X[f'{self.datetime_column}_weekday'] = dt.dt.weekday
        if 'hour' in self.features:
            X[f'{self.datetime_column}_hour'] = dt.dt.hour
        if 'minute' in self.features:
            X[f'{self.datetime_column}_minute'] = dt.dt.minute
        if 'second' in self.features:
            X[f'{self.datetime_column}_second'] = dt.dt.second
        if 'quarter' in self.features:
            X[f'{self.datetime_column}_quarter'] = dt.dt.quarter
        if 'dayofyear' in self.features:
            X[f'{self.datetime_column}_dayofyear'] = dt.dt.dayofyear
        if 'is_weekend' in self.features:
            X[f'{self.datetime_column}_is_weekend'] = dt.dt.weekday >= 5

        return X

class TargetCategoryClusterer(BaseEstimator, TransformerMixin):
    """
    Clusters a categorical feature based on the statistical profile of a target variable,
    using Bayesian Gaussian Mixture with ELBO and Silhouette-Score for optimization.

    The resulting cluster label is mapped back to each row as a new feature.

    Parameters
    ----------
    cat_feature : str
        Name of the categorical column to cluster (e.g. 'departure_airport').
    data_feature : str
        Name of the target/continuous feature used for clustering.
    n_clusters_max : int, default=10
        Maximum number of clusters to try.
    agg_funcs : list of str, default=['mean', 'std', 'median', 'count']
        Aggregation functions applied to the target variable per category.
    scaler : sklearn transformer, default=RobustScaler()
        Scaler applied to aggregated features before clustering.
    elbo_threshold : float, default=0.01
        Minimum cluster weight to keep a cluster (based on ELBO).
    silhouette_optimize : bool, default=True
        Whether to use Silhouette-Score to refine cluster selection.
    new_feature_name : str or None, default=None
        Name of the resulting cluster feature. If None, the original categorical column is overwritten.
    random_state : int, default=42
        Random state used for reproducibility.
    """


    def __init__(self, cat_feature: str, data_feature: str, n_clusters_max: int = 10, agg_funcs=None, scaler=None,
                 elbo_threshold: float = 0.01, new_feature_name: str | None = None, random_state: int = 42, ):
        self.cat_feature = cat_feature
        self.data_feature = data_feature
        self.n_clusters_max = n_clusters_max
        self.agg_funcs = agg_funcs or ['mean', 'std', 'median', 'count']
        self.scaler = scaler or RobustScaler()
        self.elbo_threshold = elbo_threshold
        self.new_feature_name = new_feature_name
        self.random_state = random_state

        self.clusterer_ = None
        self.categories_ = None

    def fit(self, X: pd.DataFrame, y = None):
        df = X[[self.cat_feature, self.data_feature]].copy()
        category_stats = df.groupby(self.cat_feature)[self.data_feature].agg(self.agg_funcs).fillna(0)
        X_scaled = self.scaler.fit_transform(category_stats)

        best_score = -1
        best_labels = None

        for n in range(2, self.n_clusters_max + 1):
            self.clusterer_ = BayesianGaussianMixture(
                n_components=n,
                weight_concentration_prior_type='dirichlet_process',
                random_state=self.random_state
            )
            self.clusterer_.fit(X_scaled)
            labels = self.clusterer_.predict(X_scaled)

            keep_clusters = [i for i, w in enumerate(self.clusterer_.weights_) if w > self.elbo_threshold]
            labels_filtered = np.array([l if l in keep_clusters else -1 for l in labels])

            valid_mask = labels_filtered != -1
            if len(set(labels_filtered[valid_mask])) > 1:
                score = silhouette_score(X_scaled[valid_mask], labels_filtered[valid_mask])
                if score > best_score:
                    best_score = score
                    best_labels = labels_filtered


        category_stats['cluster'] = best_labels
        self.categories_ = category_stats['cluster'].to_dict()

        return self

    def transform(self, X: pd.DataFrame):
        X_out = X.copy()

        feature_name = self.new_feature_name or self.cat_feature

        X_out[feature_name] = X_out[self.cat_feature].map(self.categories_).fillna(-1).astype("category")

        return X_out

class DBSCANGeoClustererCentroid(BaseEstimator, TransformerMixin):
    """
    Clusters geographical points (latitude, longitude) using DBSCAN with Haversine distance,
    and maps new points to the nearest cluster centroid.

    Parameters
    ----------
    lat_col : str
        Name of the latitude column.

    lon_col : str
        Name of the longitude column.

    eps_km : float, default=100
        Maximum distance in kilometers to consider points as neighbors in DBSCAN.

    min_samples : int, default=2
        Minimum number of points to form a cluster in DBSCAN.

    new_feature_name : str or None, default=None
        Name of the resulting cluster feature. If None, overwrites the latitude column.

    max_distance_km : float or None, default=None
        Optional: maximum distance to assign a cluster. Points farther than this will be labeled -1.
    """

    def __init__(self, lat_col, lon_col, eps_km=100, min_samples=2, new_feature_name=None, max_distance_km=None):
        self.lat_col = lat_col
        self.lon_col = lon_col
        self.eps_km = eps_km
        self.min_samples = min_samples
        self.new_feature_name = new_feature_name
        self.max_distance_km = max_distance_km
        self.earth_radius_km = 6371.0088  # mean Earth radius

    def fit(self, X, y=None):
        df = X[[self.lat_col, self.lon_col]].copy()
        coords_rad = np.radians(df.to_numpy())

        self.clusterer_ = DBSCAN(
            eps=self.eps_km / self.earth_radius_km,
            min_samples=self.min_samples,
            metric='haversine'
        )
        self.clusterer_.fit(coords_rad)
        self.labels_ = self.clusterer_.labels_

        clusters = np.unique(self.labels_)
        clusters = clusters[clusters != -1]

        self.centroids_ = {}
        for c in clusters:
            self.centroids_[c] = coords_rad[self.labels_ == c].mean(axis=0)

        self.tree_ = BallTree(np.array(list(self.centroids_.values())), metric='haversine')
        self.cluster_keys_ = list(self.centroids_.keys())

        return self

    def transform(self, X):
        X_out = X.copy()
        feature_name = self.new_feature_name or self.lat_col
        coords_rad = np.radians(X_out[[self.lat_col, self.lon_col]].to_numpy())

        dist, idx = self.tree_.query(coords_rad, k=1)
        dist_km = dist.flatten() * self.earth_radius_km
        nearest_cluster = [self.cluster_keys_[i] for i in idx.flatten()]

        if self.max_distance_km is not None:
            nearest_cluster = [
                c if d <= self.max_distance_km else -1
                for c, d in zip(nearest_cluster, dist_km)
            ]

        X_out[feature_name] = nearest_cluster
        return X_out[[feature_name]]