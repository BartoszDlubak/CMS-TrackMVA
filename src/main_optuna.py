import optuna
import pandas as pd
import numpy as np
import yaml
from pathlib import Path

from preprocess.preprocess import split_and_norm
from build import get_trainer
from tuning.optuna_configs import get_optuna_config  
from utils import load_experiment_config

study_name = 'study123'
db_path = 'dbs/optuna_new'
n_trials = 10
n_startup_trials=5
n_warmup_steps=150


# 0. Creating relevant objects
preprocess_cfg, ml_cfg = load_experiment_config()
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
# X_test, y_test = d['test'][0]


def objective(trial):
    # overwrite config with optuna suggestions
    tuned_cfg = get_optuna_config(ml_cfg, trial)
    trainer = get_trainer(tuned_cfg)

    trainer.train(X_train, X_valid, y_train, y_valid, callbacks=[])
    val_score = trainer.get_loss(X_valid, y_valid)
    return val_score

if __name__ == '__main__':
    study = optuna.create_study(
        study_name=study_name,
        direction='minimize',
        storage=f'sqlite:///{db_path}', 
        load_if_exists=True,
        pruner=optuna.pruners.MedianPruner(n_startup_trials=n_startup_trials, n_warmup_steps=n_warmup_steps)
    )
    study.optimize(objective, n_trials=n_trials)

    print('Best params: ', study.best_params)
    print('Best RMSE: ', study.best_value)