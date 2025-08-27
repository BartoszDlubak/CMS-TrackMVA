import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold

class DataSplitter():
    def __init__(self, seed: int, split_ratio: float, folds: int, folds_used: int):
        self.seed = seed
        self.split_ratio = split_ratio
        self.folds = folds
        self.folds_used = folds_used
        self._is_input_valid()
        
        self.output_train_key = 'train'
        self.output_test_key = 'test'
        
    def _is_input_valid(self):
        #Testing dtypes:
        assert isinstance(self.seed, int), f' The input: \'seed\' must be an integer.'
        assert isinstance(self.split_ratio, float), f' The input: \'split_ratio\' must be a float.'
        assert isinstance(self.folds, int), f' The input: \'folds\' must be an integer.'
        assert isinstance(self.folds_used, int), f' The input: \'folds_used\' must be an integer.'
        
        if ((self.split_ratio < 0) | (self.split_ratio > 1)):
            raise ValueError(' The input: \'split_ratio\' must be > 0 and < 1.')
        if (self.folds_used > self.folds): 
            raise ValueError('The number of folds used must be smaller than the number of folds')
        
    def split(self, X: pd.DataFrame, y: np.ndarray) -> tuple: 
        
        test_frac = 1 - self.split_ratio
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_frac, random_state=self.seed, shuffle=True)
        folds = self._cv_split(X_train, y_train)
        d = {
            self.output_train_key: folds,
            self.output_test_key: [(X_test, y_test)]
        }
        return d
        
    def _cv_split(self, X: pd.DataFrame, y: np.ndarray = None) -> list:
        kf = KFold(n_splits=self.folds, shuffle=True, random_state=self.seed)
        folds = []
        for i, indices in enumerate(kf.split(X)): #kf.split() return the indices for each fold
            if i >= self.folds_used:
                break
            train_idx, val_idx = indices
            folds.append(
                (
                X.iloc[train_idx], X.iloc[val_idx],
                None if y is None else y.iloc[train_idx],
                None if y is None else y.iloc[val_idx]
                )
                )
        return folds
