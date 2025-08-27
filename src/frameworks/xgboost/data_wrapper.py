import pandas as pd
import xgboost as xgb

class XGBWrapper:
    def __init__(self, X, y=None):
        if isinstance(X, (pd.DataFrame, pd.Series)):
            X = X.values
        if y is not None and isinstance(y, (pd.DataFrame, pd.Series)):
            y = y.values
        self.dmatrix = xgb.DMatrix(X, label=y)

    def wrap(self):
        return self.dmatrix