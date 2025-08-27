from .mlp import MLP
# add more as needed

def build_model(model_cfg):
    model_type = model_cfg['type'].lower()

    if model_type == 'mlp':
        return MLP(
            input_dim=model_cfg['input_dim'],
            hidden_dims=model_cfg['hidden_dims'],
            output_dim=model_cfg['output_dim'],
            dropout=model_cfg.get('dropout', 0.0),
        )

    else:
        raise ValueError(f'Unknown model type {model_type}')