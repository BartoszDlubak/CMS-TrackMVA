import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader


class Dataset(Dataset):
    def __init__(self, X, y):
        if isinstance(X, (pd.DataFrame, pd.Series)):
            X = X.values
        if y is not None and isinstance(y, (pd.DataFrame, pd.Series)):
            y = y.values
        self.features = torch.tensor(X, dtype=torch.float32)
        self.labels = torch.tensor(y, dtype=torch.float32)
        
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]
    

class TorchDataWrapper():
    def __init__(self, batch_size, shuffle):
        self.batch_size = batch_size
        self.shuffle = shuffle
    
    def __call__(self, X, y):
        dataset = Dataset(X,y)
        dataloader = DataLoader(dataset, self.batch_size, self.shuffle)
        return dataloader
    