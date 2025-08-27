def suggest(trial, ml_cfg):
    ml_cfg = ml_cfg.copy()

    num_layers = trial.suggest_int("num_layers", 2, 5)
    hidden_dims = []
    for i in range(num_layers):
        hidden_dims.append(
            trial.suggest_int(f"hidden_dim_{i}", 32, 512, step=32)
        )
        
    ml_cfg["model"]["hidden_dims"] = hidden_dims
    ml_cfg["model"]["num_layers"] = num_layers
    ml_cfg["model"]["dropout"] = trial.suggest_float("dropout", 0.0, 0.5)
    ml_cfg["train"]["lr"] = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
    ml_cfg["train"]["weight_decay"] = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
    # ml_cfg["train"]["epochs"] = trial.suggest_int("epochs", 5, 50)

    return ml_cfg