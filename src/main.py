import pandas as pd
import yaml
from pathlib import Path
import numpy as np

from preprocess.preprocess import split_and_norm

from build import get_trainer
from utils import load_experiment_config


# 0. Creating relevant objects
preprocess_cfg, ml_cfg = load_experiment_config('configs/experiment.yaml')
save_path = 'saves/xgb.pkl'
ml_cfg['train']['save_path'] = save_path
trainer = get_trainer(ml_cfg)

# 1. Loading Data
data_path = Path('data/mva.parquet')
data = pd.read_parquet(data_path)

cols = preprocess_cfg['normalise']['features'].keys()
X = data[cols]
y = data['match']

print('Input Columns: ', X.columns)

# 2. Preprocessing
d = split_and_norm(X, y, preprocess_cfg)
folds = d['train']
print('Number of folds: ', len(folds))

X_train, X_valid, y_train, y_valid = folds[0]

X_train = X_train.astype(np.float32).to_numpy()
y_train = y_train.astype(np.float32).to_numpy()
X_valid = X_valid.astype(np.float32).to_numpy()
y_valid = y_valid.astype(np.float32).to_numpy()

# X_test, y_test = d['test'][0]
# print('new: ')
# print(X_train['col1'][0:10], X_train['col2'][0:10], X_train['col3'][0:10])


# 4. Training
trainer.train(X_train, X_valid, y_train, y_valid)
