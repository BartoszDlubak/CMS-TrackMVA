def suggest(trial, ml_cfg):
    ml_cfg = ml_cfg.copy()

    num_layers = trial.suggest_int("num_layers", 2, 5)
    hidden_dims = []
    for i in range(num_layers):
        hidden_dims.append(
            trial.suggest_int(f"hidden_dim_{i}", 32, 512, step=32)
        )
        
    ml_cfg["model"]["hidden_dims"] = hidden_dims
    ml_cfg["model"]["dropout"] = trial.suggest_float("dropout", 0.0, 0.02)
    ml_cfg["train"]["lr"] = trial.suggest_float("lr", 1e-4, 1e-1, log=True)
    ml_cfg["train"]["weight_decay"] = trial.suggest_float("weight_decay", 1e-6, 1e-5, log=True)
    # ml_cfg["train"]["epochs"] = trial.suggest_int("epochs", 5, 50)

    return ml_cfg