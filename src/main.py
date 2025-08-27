import pandas as pd
import numpy as np
import yaml
from pathlib import Path

from preprocess.preprocess import split_and_norm

from build import get_trainer


def load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def load_experiment_config(path="configs/experiment.yaml"):
    cfg = load_yaml(path)
    preprocess_cfg = load_yaml(f"configs/{cfg['preprocess']}")
    ml_cfg = load_yaml(f"configs/{cfg['model']}")
    return preprocess_cfg, ml_cfg


# 0. Creating relevant objects
preprocess_cfg, ml_cfg = load_experiment_config()
trainer = get_trainer(ml_cfg)

# 1. Loading Data
data_path = Path('data/mva.parquet')
data = pd.read_parquet(data_path)

cols = preprocess_cfg['features'].keys()
X = data[cols]
y = data['match']
print('Input Columns: ', X.columns)

# 2. Preprocessing
d = split_and_norm(X, y, preprocess_cfg)
folds = d['train']
print('Number of folds: ', len(folds))

X_train, X_valid, y_train, y_valid = folds[0]
# X_test, y_test = d['test'][0]
# print('new: ')
# print(X_train['col1'][0:10], X_train['col2'][0:10], X_train['col3'][0:10])


# 4. Training
trainer.train(X_train, X_valid, y_train, y_valid)
