from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.metrics import accuracy_score, mean_squared_error

from Utils.Styling import RES, BLU_L

class LayeredDelayModel(BaseEstimator, RegressorMixin):
    """
    A layered machine learning model for predicting delays, handling zero-inflation.

    This model uses a two-stage approach:
    1. A classifier predicts the probability that a flight will be delayed (target > 0).
    2. A regressor predicts the expected delay given that the flight is delayed (target | target > 0).
    The final prediction is computed as:
        E(delay) = P(delay > 0) * E(delay | delay > 0)

    Parameters
    ----------
    clf_params : dict, optional
        Parameters for the CatBoostClassifier used in the first stage.
    reg_params : dict, optional
        Parameters for the CatBoostRegressor used in the second stage.
    cat_features_clf : list of str, optional
        List of categorical feature names for the classifier.
    cat_features_reg : list of str, optional
        List of categorical feature names for the regressor.
    num_features_clf : list of str, optional
        List of numerical feature names for the classifier.
    num_features_reg : list of str, optional
        List of numerical feature names for the regressor.

    Methods
    -------
    fit(X, y)
        Fit both classifier and regressor to the training data.
    predict(X)
        Return the layered prediction for the input data.
    """

    def __init__(self,
                 clf_params=None,
                 reg_params=None,
                 cat_features_clf=None,
                 cat_features_reg=None,
                 num_features_clf=None,
                 num_features_reg=None):
        self.clf_params = clf_params if clf_params else {}
        self.reg_params = reg_params if reg_params else {}
        self.cat_features_clf = cat_features_clf
        self.cat_features_reg = cat_features_reg
        self.num_features_clf = num_features_clf
        self.num_features_reg = num_features_reg
        self.clf_ = None
        self.reg_ = None

    def fit(self, X, y):
        y_cls = (y > 0).astype(int)
        X_clf = X[self.cat_features_clf + self.num_features_clf]
        self.clf_ = CatBoostClassifier(**self.clf_params)
        self.clf_.fit(X_clf, y_cls, cat_features=self.cat_features_clf, verbose=100)

        mask = y > 0
        X_reg = X.loc[mask, self.cat_features_reg + self.num_features_reg]
        y_reg = y.loc[mask]
        self.reg_ = CatBoostRegressor(**self.reg_params)
        self.reg_.fit(X_reg, y_reg, cat_features=self.cat_features_reg, verbose=100)
        return self

    def predict(self, X):
        X_clf = X[self.cat_features_clf + self.num_features_clf]
        p_delay = self.clf_.predict_proba(X_clf)[:, 1]

        X_reg = X[self.cat_features_reg + self.num_features_reg]
        delay_cond = self.reg_.predict(X_reg).clip(0)

        return p_delay * delay_cond

    def evaluate(self, X_test, y_test):
        """
        Evaluate the layered model on test data.

        Parameters
        ----------
        X_test : pd.DataFrame
            Test features.
        y_test : pd.Series or np.array
            True target values.

        Prints
        ------
        Classifier accuracy on all test data.
        Regressor RMSE on test samples with target > 0.
        """

        print(f"{BLU_L}=== Layered Delay Model Evaluation ==={RES}")

        y_cls_test = (y_test > 0).astype(int)
        X_clf_test = X_test[self.cat_features_clf + self.num_features_clf]
        y_cls_pred = self.clf_.predict(X_clf_test)
        acc_test = accuracy_score(y_cls_test, y_cls_pred)
        print(f"Classifier Accuracy: {acc_test:.4f}")

        mask = y_test > 0
        X_reg_test = X_test.loc[mask, self.cat_features_reg + self.num_features_reg]
        y_reg_test = y_test.loc[mask]
        y_reg_pred = self.reg_.predict(X_reg_test)
        rmse_test = mean_squared_error(y_reg_test, y_reg_pred, squared=False)
        print(f"Regressor RMSE (delay>0): {rmse_test:.2f}")
