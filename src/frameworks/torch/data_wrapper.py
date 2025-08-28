import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class Dataset(Dataset):
    def __init__(self, X, y, L_FP, L_FN):
        if isinstance(X, (pd.DataFrame, pd.Series)):
            X = X.values
        if y is not None and isinstance(y, (pd.DataFrame, pd.Series)):
            y = y.values
        
        weights = compute_weights(y, L_FP, L_FN)
        self.weights = torch.tensor(weights, dtype=torch.float32)
        self.features = torch.tensor(X, dtype=torch.float32)
        self.labels = torch.tensor(y, dtype=torch.long)
        
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx], self.weights[idx]
    

class TorchDataWrapper():
    def __init__(self, batch_size, shuffle, L_FP, L_FN):
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.L_FP = L_FP
        self.L_FN = L_FN
    
    def __call__(self, X, y):
        dataset = Dataset(X, y, self.L_FP, self.L_FN)
        print(len(dataset))
        dataloader = DataLoader(dataset, self.batch_size, self.shuffle)
        return dataloader
    

def compute_weights(y, L_FP, L_FN):
    y = np.asarray(y)
    w1 = y.mean()
    w0 = 1.0 - w1
    w_pos = L_FN / w1
    w_neg = L_FP / w0
    c = 1.0 / (w_pos + w_neg)
    return np.where(y == 1, w_pos * c, w_neg * c)