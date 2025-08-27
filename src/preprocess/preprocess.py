import pandas as pd
import numpy as np
from .datasplit import DataSplitter
from .normalise import Normaliser


def split_and_norm(X, y, cfg):
    # load_paths = cfg['load']
    datasplit_cfg = cfg['data_split']
    normaliser_cfg = cfg['normalise']
    
    datasplitter = DataSplitter(
        seed= datasplit_cfg['seed'],
        split_ratio=datasplit_cfg['split_ratio'],
        folds=datasplit_cfg['folds'],
        folds_used=datasplit_cfg['folds_used']
    )
    normaliser = Normaliser(
        feautre_norms=normaliser_cfg['features'],
        label_norms=normaliser_cfg['labels']
    )
    
    d = datasplitter.split(X, y)
    folds = d['train']
    for i in range(len(folds)):
        X_train, X_valid, y_train, y_valid = folds[i]
        X_train = normaliser.transform_features(X_train)
        X_valid = normaliser.transform_features(X_valid)
        folds[i] = X_train, X_valid, y_train, y_valid
     
    return d
    