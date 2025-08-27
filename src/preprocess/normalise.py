import numpy as np
import pandas as pd


class Normaliser():
    def __init__(self, feautre_norms: dict, label_norms: dict = None):
        self.feature_norms = feautre_norms #dict of feauture cols (keys) and normalisations (vals)
        self.labels_norms = label_norms #dict of label cols (keys) and normalisations (vals)
        self.norms = {
            'standard': StandardNorm,
            'log_standard': LogStandardNorm,
            'none': 0,
            'remove': 0
        }
        
    def transform_features(self, df):
        df = self._transform(df, self.feature_norms)
        return df
    
    def transform_labels(self, df):
        df = self._transform(df, self.labels_norms)    
        return df 
    
    def _transform(self, df: pd.DataFrame, norms_dict):
        df = df.copy()
        to_remove = 'remove'
        to_pass = 'none'
        
        for col, norm_name in norms_dict.items():
            if norm_name not in self.norms:
                raise KeyError(f'Normalisation {norm_name} is invalid')
            if col not in df.columns:
                raise KeyError(f'Column \'{col}\' does not exist.')
            
            if norm_name == to_remove:
                df.drop(columns = col, inplace = True)
                continue
            elif norm_name == to_pass:
                continue
            else:
                values = df[col].values
                norm_class = self.norms[norm_name]
                norm = norm_class(values)
                df[col] = norm.normalise(values)
        return df
    
    
class StandardNorm():
    def __init__(self, x: np.ndarray):
        self.mean = x.mean()
        self.std = x.std()
        self.params = {
            'mean': self.mean,
            'std': self.std
        }

    def normalise(self, x):
        return (x - self.mean) / self.std
                
    def unnormalise(self, x):
        return self.std * x + self.mean
    

class LogStandardNorm():
    def __init__(self, x: np.ndarray):
        self.mean = np.log(x).mean()
        self.std = np.log(x).std()
        self.params = {
            'log_mean': self.mean,
            'log_std': self.std
        }

    def normalise(self, x):
        x = np.log(x)
        return (x - self.mean) / self.std
                
    def unnormalise(self, x):
        x = self.std * x + self.mean
        return np.exp(x)
        
        
        
        