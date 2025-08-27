from tuning.spaces import xgboost, torch_mlp

def get_optuna_config(ml_cfg, trial):
    framework = ml_cfg["framework"].lower()

    if framework == "torch":
        model_type = ml_cfg["model"]["type"].lower()
        if model_type == "mlp":
            return torch_mlp.suggest(trial, ml_cfg)
        elif model_type == 'cnn':
            pass
    elif framework == "xgboost":
        return xgboost.suggest(trial, ml_cfg)
    else:
        raise ValueError(f"No Optuna config available for framework={framework}, model={model_type}")