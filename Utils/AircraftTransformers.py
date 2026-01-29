import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class AircraftClusterTransformer(BaseEstimator, TransformerMixin):
    """
    Transformer to map aircraft codes to categories.

    Parameters
    ----------
    column : str
        Name of the column containing aircraft codes.
    new_column : str or None, default=None
        Name of the new column to store categories.
        If None, the original column will be overwritten.
    """

    def __init__(self, column, new_column=None):
        self.column = column
        self.new_column = new_column or column

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        def map_aircraft(ac):
            if pd.isna(ac):
                return "other"
            ac = str(ac)
            if "AT7" in ac:
                return "turboprop"
            if "CR9" in ac:
                return "regional_jet"
            if any(k in ac for k in ["31A", "31B", "319", "320", "32A", "321", "736", "733", "738"]):
                return "narrow_body"
            if "332" in ac:
                return "wide_body"
            if "M87" in ac:
                return "legacy"
            return "other"

        X[self.new_column] = X[self.column].apply(map_aircraft)
        return X