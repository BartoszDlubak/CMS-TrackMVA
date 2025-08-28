def suggest(trial, ml_cfg):
    ml_cfg = ml_cfg.copy()
    ml_cfg['train']["params"].update({
        'max_depth': trial.suggest_int('max_depth', 3,20),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.5, log=True),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.7, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 5),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-4, 1e-1, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-2, 0.3, log=True),
        'gamma': trial.suggest_float('gamma', 0.01, 1.0, log = True),
    })
    
    return ml_cfg